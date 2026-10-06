//! Dedicated read-only comparison routes on the existing bounded loopback transport.
use crate::{
    api,
    comparison::Comparison,
    headroom, require,
    server::{self, Query},
    Result,
};
use serde_json::{json, Value};
use std::{
    io::Write,
    net::{TcpListener, TcpStream},
    sync::{
        mpsc::{sync_channel, SyncSender},
        Arc,
    },
    time::Instant,
};
enum Job {
    Calibrate(usize),
    Tile(TcpStream, Query),
}
fn code(e: &dyn std::fmt::Display) -> u16 {
    if e.to_string().contains("not ready") {
        503
    } else {
        400
    }
}
pub fn serve(state: Arc<Comparison>, port: u16) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let port = listener.local_addr()?.port();
    let (sender, receiver) = sync_channel::<Job>(8);
    let worker = state.clone();
    std::thread::Builder::new().name("atlas-comparison-numeric".into()).stack_size(2*1024*1024).spawn(move||{
        for job in receiver {match job {
            Job::Calibrate(id)=>{if let Err(e)=worker.calibrate_one(id){eprintln!("Comparison calibration paused: {e}");}},
            Job::Tile(socket,q)=>{
                if server::disconnected(&socket){continue}
                let start=Instant::now();let result=(||{worker.tile(q.int("tensor","0")?,q.get("quantity","delta"),q.get("mapping","linear"),q.int("level","0")?.try_into()?,q.int("x","0")?,q.int("y","0")?)})();
                match result {Ok((png,cached,m))=>server::reply(socket,200,"image/png",&png,&format!("X-Atlas-Factor: {}\r\nX-Atlas-Cache: {}\r\nX-Atlas-Seconds: {:.6}\r\nX-Atlas-Source-Bytes: {}\r\nX-Atlas-Coordinate-Space: checkpoint-comparison-v1\r\nX-Atlas-Inference-Editable: false\r\nX-Atlas-Comparison-Identity: {}\r\nX-Atlas-Source-A-Identity: {}\r\nX-Atlas-Source-B-Identity: {}\r\n",m.factor,if cached{"hit"}else{"miss"},start.elapsed().as_secs_f64(),m.source_bytes_read,worker.identity,worker.a.identity,worker.b.identity)),Err(e)=>server::error(socket,code(&e),e)}
            }
        }}
    })?;
    let (dispatch_sender, dispatch_receiver) =
        sync_channel::<server::CompletedRequest>(server::DISPATCH_CAPACITY);
    let dispatch_state = state.clone();
    std::thread::Builder::new()
        .name("atlas-comparison-dispatch".into())
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            for (socket, raw) in dispatch_receiver {
                dispatch(&dispatch_state, &sender, socket, &raw, port);
            }
        })?;
    println!(
        "{}",
        json!({"listening":format!("http://127.0.0.1:{port}"),"comparison_identity":state.identity,"coordinate_space":crate::comparison::VERSION,"inference_editable":false,"tensors":state.pairs.len(),"peak_rss_mib":crate::peak_rss_mib()})
    );
    std::io::stdout().flush()?;
    server::run_transport(listener, dispatch_sender)
}
// Hold the progress lock across queue admission so a fast worker cannot publish
// a failure just before the dispatcher clears the prior request's error.
fn queue_calibration(state: &Comparison, sender: &SyncSender<Job>, id: usize) -> bool {
    let mut progress = state.progress.lock().unwrap();
    if sender.try_send(Job::Calibrate(id)).is_err() {
        return false;
    }
    progress.error = None;
    true
}

fn dispatch(
    state: &Comparison,
    sender: &SyncSender<Job>,
    socket: TcpStream,
    raw: &[u8],
    port: u16,
) {
    let request = server::parse_request(raw, port).and_then(|r| {
        headroom()?;
        Ok(r)
    });
    let (method, path, q, headers) = match request {
        Ok(r) => r,
        Err(e) => {
            server::error(socket, 400, e);
            return;
        }
    };
    if !q.get(api::parameter::COMPARISON_IDENTITY, "").is_empty()
        && q.get(api::parameter::COMPARISON_IDENTITY, "") != state.identity
    {
        server::error(socket, 400, "Comparison identity changed; refresh required");
        return;
    }
    if method == "POST" && path == api::comparison::CALIBRATION {
        reply_calibration(state, sender, socket, &q, &headers);
        return;
    }
    if method != "GET" {
        server::error(
            socket,
            400,
            "Only GET and local comparison calibration POST are supported",
        );
        return;
    }
    if path == api::comparison::TILE {
        enqueue_tile(sender, socket, q);
        return;
    }
    let result = read_api(state, &path, &q);
    if let Some(result) = result {
        match result {
            Ok(v) => server::json_reply(socket, 200, v),
            Err(e) => server::error(socket, code(&e), e),
        }
        return;
    }
    reply_asset(socket, &path)
}

fn reply_calibration(
    state: &Comparison,
    sender: &SyncSender<Job>,
    socket: TcpStream,
    q: &Query,
    headers: &std::collections::BTreeMap<String, String>,
) {
    let result = (|| -> Result<usize> {
        require(
            headers.get("x-atlas-local").map(String::as_str) == Some("1"),
            "Local action header required",
        )?;
        require(
            q.get("all", "0") != "1",
            "Comparison calibration is explicitly tensor-scoped",
        )?;
        state.check()?;
        let id = q.int("tensor", "")?;
        state.pair(id)?;
        Ok(id)
    })();
    match result {
        Ok(id) => {
            if state.scales(id).is_some() {
                let mut v = state.identity_metadata();
                v["complete"] = json!(true);
                server::json_reply(socket, 200, v);
            } else if queue_calibration(state, sender, id) {
                let mut v = state.identity_metadata();
                v["queued"] = json!(id);
                server::json_reply(socket, 202, v);
            } else {
                server::error(socket, 503, "Numeric queue full; retry shortly");
            }
        }
        Err(e) => server::error(socket, 400, e),
    }
}

fn enqueue_tile(sender: &SyncSender<Job>, socket: TcpStream, q: Query) {
    if let Err(e) = sender.try_send(Job::Tile(socket, q)) {
        let (std::sync::mpsc::TrySendError::Full(job)
        | std::sync::mpsc::TrySendError::Disconnected(job)) = e;
        if let Job::Tile(s, _) = job {
            server::error(s, 503, "Numeric queue full; retry shortly")
        }
    }
}

fn read_api(state: &Comparison, path: &str, q: &Query) -> Option<Result<Value>> {
    match path {
        api::comparison::MODEL => Some(state.model()),
        api::comparison::VIEW => Some((|| {
            state.view(
                q.int(api::parameter::TENSOR, "0")?,
                q.get(api::parameter::LEFT, "a"),
                q.get(api::parameter::RIGHT, "b"),
                q.get(api::parameter::MAPPING, "linear"),
            )
        })()),
        api::comparison::INSPECT => Some((|| {
            state.inspect(
                q.int(api::parameter::TENSOR, "0")?,
                q.int(api::parameter::ROW, "0")?,
                q.int(api::parameter::COL, "0")?,
            )
        })()),
        _ => None,
    }
}

fn reply_asset(socket: TcpStream, path: &str) {
    let asset: Option<(&str, &[u8])> = match path {
        "/" | "/comparison.html" => Some((
            "text/html; charset=utf-8",
            include_bytes!("../web/comparison.html"),
        )),
        "/comparison.js" => Some(("text/javascript", include_bytes!("../web/comparison.js"))),
        "/comparison.css" => Some(("text/css", include_bytes!("../web/comparison.css"))),
        "/vendor/openseadragon.min.js" => Some((
            "text/javascript",
            include_bytes!("../web/vendor/openseadragon.min.js"),
        )),
        "/vendor/OpenSeadragon-LICENSE.txt" => Some((
            "text/plain",
            include_bytes!("../web/vendor/OpenSeadragon-LICENSE.txt"),
        )),
        _ => None,
    };
    if let Some((mime, body)) = asset {
        server::reply(socket, 200, mime, body, "")
    } else {
        server::error(socket, 404, "Not found on read-only comparison server")
    }
}

#[cfg(test)]
mod dispatch_vectors {
    //! Tiny comparison dispatch vectors; no listener loop or numeric worker.
    use super::*;
    use std::{
        io::Read,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::Duration,
    };

    struct Fixture {
        state: Comparison,
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "atlas-comparison-dispatch-vectors-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let header = br#"{"weights":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}}"#;
            for (name, payload) in [("a", [0x80, 0x3f, 0x80, 0xbf]), ("b", [0, 0x40, 0, 0xc0])] {
                let model = root.join(name);
                std::fs::create_dir_all(&model).unwrap();
                let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
                bytes.extend_from_slice(header);
                bytes.extend_from_slice(&payload);
                std::fs::write(model.join("tiny.safetensors"), bytes).unwrap();
            }
            let state =
                Comparison::open(&root.join("a"), &root.join("b"), &root.join("cache")).unwrap();
            Self { state, root }
        }

        fn request(&self, path: &str) -> (String, Value) {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let address = listener.local_addr().unwrap();
            let mut client = TcpStream::connect_timeout(&address, Duration::from_secs(3)).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let (socket, _) = listener.accept().unwrap();
            let (sender, _receiver) = sync_channel(8);
            let raw = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\n\r\n");
            dispatch(&self.state, &sender, socket, raw.as_bytes(), address.port());
            let mut bytes = Vec::new();
            client.read_to_end(&mut bytes).unwrap();
            assert!(bytes.len() < 64 * 1024, "tiny comparison response bound");
            let response = String::from_utf8(bytes).unwrap();
            let (head, body) = response.split_once("\r\n\r\n").unwrap();
            (
                head.lines().next().unwrap().to_owned(),
                serde_json::from_str(body).unwrap(),
            )
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn model_route_keeps_comparison_catalog_and_coordinate_space() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (status, body) = fixture.request("/api/comparison/model");
        assert_eq!(status, "HTTP/1.1 200 OK", "comparison model route status");
        assert_eq!(body["api_version"], 1);
        assert_eq!(body["coordinate_space"], "checkpoint-comparison-v1");
        assert_eq!(body["inference_editable"], false);
        assert_eq!(body["compatibility"]["tensor_count"], 1);
        assert_eq!(body["catalog"][0]["name"], "weights");
        assert_eq!(body["catalog"][0]["calibration_complete"], false);
        assert_eq!(body["comparison_identity"], fixture.state.identity);
    }

    #[test]
    fn inspect_route_keeps_selected_originals_and_difference_direction() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (status, body) = fixture.request("/api/comparison/inspect?tensor=0&row=0&col=1");
        assert_eq!(status, "HTTP/1.1 200 OK", "comparison inspect route status");
        assert_eq!(body["native_indices"], json!([0, 1]));
        assert_eq!(body["originals"]["a"]["raw_exact"], "-1");
        assert_eq!(body["originals"]["b"]["raw_exact"], "-2");
        assert_eq!(body["difference"]["value"], -1.0);
        assert_eq!(body["difference"]["direction"], "B-A");
        assert_eq!(body["difference"]["derived"], true);
        assert_eq!(body["inference_editable"], false);
    }

    #[test]
    fn unknown_asset_keeps_comparison_not_found_wording() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (status, body) = fixture.request("/missing-fixture-asset");
        assert_eq!(
            status, "HTTP/1.1 404 Not Found",
            "comparison missing asset status"
        );
        assert_eq!(
            body,
            json!({"api_version":1,"error":"Not found on read-only comparison server"}),
            "comparison missing asset body"
        );
    }

    impl Fixture {
        // One owned socket pair, no listener loop or numeric worker. The closed
        // receiver holds queue refusals without leaving an enqueued socket open.
        fn declaration_request(&self, method: &str, path: &str, header: &str) -> (String, Value) {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let address = listener.local_addr().unwrap();
            let mut client = TcpStream::connect_timeout(&address, Duration::from_secs(3)).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let (socket, _) = listener.accept().unwrap();
            let (sender, receiver) = sync_channel(1);
            drop(receiver);
            let raw = format!("{method} {path} HTTP/1.1\r\nHost: {address}\r\n{header}\r\n");
            dispatch(&self.state, &sender, socket, raw.as_bytes(), address.port());
            let mut bytes = Vec::new();
            client.read_to_end(&mut bytes).unwrap();
            assert!(bytes.len() < 64 * 1024, "bounded declaration response");
            let response = String::from_utf8(bytes).unwrap();
            let (head, body) = response.split_once("\r\n\r\n").unwrap();
            (
                head.lines().next().unwrap().into(),
                serde_json::from_str(body).unwrap(),
            )
        }
    }

    #[test]
    fn declaration_routes_keep_identity_method_and_queue_priority() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        for (method, path, header, status, message) in [
            (
                "POST",
                "/api/comparison/model",
                "",
                "HTTP/1.1 400 Bad Request",
                "Only GET and local comparison calibration POST are supported",
            ),
            (
                "POST",
                "/api/comparison/calibrate?comparison_identity=stale",
                "",
                "HTTP/1.1 400 Bad Request",
                "Comparison identity changed; refresh required",
            ),
            (
                "POST",
                "/api/comparison/calibrate?tensor=0",
                "",
                "HTTP/1.1 400 Bad Request",
                "Local action header required",
            ),
            (
                "POST",
                "/api/comparison/calibrate?all=1",
                "X-Atlas-Local: 1\r\n",
                "HTTP/1.1 400 Bad Request",
                "Comparison calibration is explicitly tensor-scoped",
            ),
            (
                "POST",
                "/api/comparison/calibrate?tensor=0",
                "X-Atlas-Local: 1\r\n",
                "HTTP/1.1 503 Service Unavailable",
                "Numeric queue full; retry shortly",
            ),
            (
                "GET",
                "/api/comparison/tile?tensor=0",
                "",
                "HTTP/1.1 503 Service Unavailable",
                "Numeric queue full; retry shortly",
            ),
            (
                "GET",
                "/api/comparison/view?tensor=bad",
                "",
                "HTTP/1.1 400 Bad Request",
                "invalid digit found in string",
            ),
        ] {
            let (observed, body) = fixture.declaration_request(method, path, header);
            assert_eq!(observed, status, "comparison declaration status: {path}");
            assert_eq!(
                body,
                json!({"api_version":1,"error":message}),
                "comparison declaration refusal: {path}"
            );
        }
        let (status, body) = fixture.request(&format!(
            "/api/comparison/model?comparison_identity={}&retained-extra=value",
            fixture.state.identity
        ));
        assert_eq!(
            status, "HTTP/1.1 200 OK",
            "comparison existing extra query behavior"
        );
        assert_eq!(body["comparison_identity"], fixture.state.identity);
    }
}
