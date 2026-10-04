//! Dedicated read-only comparison routes on the existing bounded loopback transport.
use crate::{
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
    if !q.get("comparison_identity", "").is_empty()
        && q.get("comparison_identity", "") != state.identity
    {
        server::error(socket, 400, "Comparison identity changed; refresh required");
        return;
    }
    if method == "POST" && path == "/api/comparison/calibrate" {
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
    if path == "/api/comparison/tile" {
        if let Err(e) = sender.try_send(Job::Tile(socket, q)) {
            let (std::sync::mpsc::TrySendError::Full(job)
            | std::sync::mpsc::TrySendError::Disconnected(job)) = e;
            if let Job::Tile(s, _) = job {
                server::error(s, 503, "Numeric queue full; retry shortly")
            }
        }
        return;
    }
    let result: Option<Result<Value>> = match path.as_str() {
        "/api/comparison/model" => Some(state.model()),
        "/api/comparison/view" => Some((|| {
            state.view(
                q.int("tensor", "0")?,
                q.get("left", "a"),
                q.get("right", "b"),
                q.get("mapping", "linear"),
            )
        })()),
        "/api/comparison/inspect" => Some((|| {
            state.inspect(
                q.int("tensor", "0")?,
                q.int("row", "0")?,
                q.int("col", "0")?,
            )
        })()),
        _ => None,
    };
    if let Some(result) = result {
        match result {
            Ok(v) => server::json_reply(socket, 200, v),
            Err(e) => server::error(socket, code(&e), e),
        }
        return;
    }
    let asset: Option<(&str, &[u8])> = match path.as_str() {
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
