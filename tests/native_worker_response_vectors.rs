//! Bounded real worker requests over two tiny BF16 tensors; no inference/browser.
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

struct Fixture {
    root: PathBuf,
    child: Option<Child>,
    address: Option<SocketAddr>,
}

struct Response {
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}
impl Response {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
    fn refusal(&self, status: u16, message: &str) {
        assert_eq!(self.status, status, "worker refusal status: {message}");
        assert_eq!(
            self.json(),
            json!({"api_version":1,"error":message}),
            "worker refusal body"
        );
    }
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-worker-vectors-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists(), "worker fixture root is new");
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let header = json!({
            "first":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]},
            "second":{"dtype":"BF16","shape":[1,2],"data_offsets":[4,8]}
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for word in [0x3f80u16, 0xc000, 0x4040, 0x4080] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        std::fs::write(model.join("tiny.safetensors"), bytes).unwrap();
        let mut fixture = Self {
            root,
            child: None,
            address: None,
        };
        let child = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
            .current_dir(&fixture.root)
            .arg("serve")
            .arg("--model")
            .arg(&model)
            .arg("--cache")
            .arg(fixture.root.join("cache"))
            .args(["--port", "0"])
            .stdout(Stdio::from(
                File::create(fixture.root.join("stdout")).unwrap(),
            ))
            .stderr(Stdio::from(
                File::create(fixture.root.join("stderr")).unwrap(),
            ))
            .spawn()
            .unwrap();
        eprintln!("worker child pid: {}", child.id());
        fixture.child = Some(child);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            assert!(
                fixture
                    .child
                    .as_mut()
                    .unwrap()
                    .try_wait()
                    .unwrap()
                    .is_none(),
                "worker server remains live during startup"
            );
            let text = std::fs::read_to_string(fixture.root.join("stdout")).unwrap();
            assert!(text.len() < 32 * 1024, "worker startup output bound");
            for line in text.lines() {
                if let Ok(value) = serde_json::from_str::<Value>(line) {
                    if let Some(url) = value["listening"].as_str() {
                        fixture.address =
                            Some(url.strip_prefix("http://").unwrap().parse().unwrap());
                        return fixture;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("worker startup deadline assertion");
    }

    fn request(&self, method: &str, path: &str) -> Response {
        let address = self.address.unwrap();
        let connected = TcpStream::connect_timeout(&address, Duration::from_secs(3));
        assert!(
            connected.is_ok(),
            "worker connection assertion: {connected:?}"
        );
        let mut socket = connected.unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let local = if method == "POST" {
            "X-Atlas-Local: 1\r\n"
        } else {
            ""
        };
        let wire = format!("{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Length: 0\r\n{local}\r\n");
        let sent = socket.write_all(wire.as_bytes());
        assert!(sent.is_ok(), "worker write assertion: {sent:?}");
        let mut bytes = Vec::new();
        let received = socket.take(64 * 1024 + 1).read_to_end(&mut bytes);
        assert!(
            received.is_ok(),
            "worker response read assertion: {received:?}"
        );
        assert!(bytes.len() <= 64 * 1024, "worker response bound assertion");
        let end = bytes.windows(4).position(|window| window == b"\r\n\r\n");
        assert!(end.is_some(), "worker response framing assertion");
        let end = end.unwrap();
        let head = std::str::from_utf8(&bytes[..end]).unwrap();
        let mut lines = head.lines();
        let status = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|line| {
                let (key, value) = line.split_once(':').unwrap();
                (key.to_ascii_lowercase(), value.trim().to_owned())
            })
            .collect::<BTreeMap<_, _>>();
        let body = bytes[end + 4..].to_vec();
        assert_eq!(
            headers["content-length"].parse::<usize>().unwrap(),
            body.len(),
            "worker complete response length"
        );
        Response {
            status,
            headers,
            body,
        }
    }

    fn wait_status(&self, complete: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut last = Value::Null;
        while Instant::now() < deadline {
            let response = self.request("GET", "/api/tensor-status?tensor=0");
            assert_eq!(response.status, 200, "worker status polling response");
            last = response.json();
            if complete(&last) {
                return last;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("worker calibration completion assertion: {last}");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let reaped = child.wait().is_ok();
            eprintln!("worker lifecycle: reaped={reaped}");
            for name in ["stdout", "stderr"] {
                let text = std::fs::read_to_string(self.root.join(name)).unwrap_or_default();
                eprintln!("worker {name}: {text}");
            }
            assert!(reaped, "worker child reap assertion");
        }
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn worker_requests_preserve_readiness_calibration_defaults_and_tile_headers() {
    let fixture = Fixture::new();
    fixture
        .request("GET", "/tile")
        .refusal(503, "Full checkpoint calibration is not ready");
    fixture
        .request("GET", "/tile?rule=tensor_linear")
        .refusal(503, "Complete selected-tensor calibration is not ready");
    fixture
        .request("GET", "/tile?tensor=bad")
        .refusal(400, "invalid digit found in string");
    fixture
        .request("GET", "/tile?rule=unknown")
        .refusal(400, "Unknown color rule");
    let queued = fixture.request("POST", "/api/calibrate?tensor=0");
    assert_eq!(queued.status, 202, "worker calibration accepted");
    assert_eq!(
        queued.json(),
        json!({"api_version":1,"queued":0}),
        "worker selected queue identity"
    );
    let local =
        fixture.wait_status(|status| status["tensor_status"]["calibration_complete"] == true);
    assert_eq!(
        local["tensor_status"]["max_abs"], 2.0,
        "worker selected exact bound"
    );
    assert_eq!(
        local["calibration_complete"], false,
        "worker selected does not complete checkpoint"
    );
    fixture
        .request("GET", "/tile")
        .refusal(503, "Full checkpoint calibration is not ready");
    let tile = fixture.request("GET", "/tile?rule=tensor_linear");
    assert_eq!(tile.status, 200, "worker calibrated tile success");
    assert_eq!(
        tile.headers["content-type"], "image/png",
        "worker tile MIME"
    );
    assert_eq!(tile.headers["x-atlas-factor"], "2", "worker factor header");
    assert_eq!(
        tile.headers["x-atlas-cache"], "miss",
        "worker first tile miss"
    );
    assert_eq!(
        tile.headers["x-atlas-source-bytes"], "4",
        "worker tiny source bytes"
    );
    assert!(
        tile.body.starts_with(b"\x89PNG\r\n\x1a\n"),
        "worker PNG framing assertion"
    );
    let cached = fixture.request("GET", "/tile?rule=tensor_linear");
    assert_eq!(cached.status, 200, "worker cached tile success");
    assert_eq!(
        cached.headers["x-atlas-cache"], "hit",
        "worker cached tile hit"
    );
    assert_eq!(cached.body, tile.body, "worker cached PNG bytes");
    fixture
        .request("GET", "/tile?rule=tensor_linear&y=1")
        .refusal(400, "Tile outside tensor");
    let queued = fixture.request("POST", "/api/calibrate?all=1");
    assert_eq!(queued.status, 202, "worker all calibration accepted");
    assert_eq!(
        queued.json(),
        json!({"api_version":1,"queued":"complete checkpoint"}),
        "worker all queue response"
    );
    let global = fixture.wait_status(|status| status["calibration_complete"] == true);
    assert_eq!(global["global_max"], 4.0, "worker exact global bound");
    assert_eq!(
        global["coverage"]["calibrated_tensors"], 2,
        "worker all tensors completed"
    );
    assert_eq!(
        fixture.request("GET", "/tile").status,
        200,
        "worker default global tile after complete calibration"
    );
}

#[test]
fn startup_diagnostic_envelope_keeps_resource_observations() {
    let fixture = Fixture::new();
    let raw = std::fs::read_to_string(fixture.root.join("stdout")).unwrap();
    let value: Value = serde_json::from_str(raw.trim()).unwrap();
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "header_bytes",
            "listening",
            "metadata_ready_seconds",
            "peak_rss_mib",
            "resources",
            "tensors"
        ]
    );
    assert_eq!(
        value["listening"],
        format!("http://{}", fixture.address.unwrap())
    );
    assert_eq!(value["tensors"], 2);
    assert_eq!(
        value["header_bytes"],
        std::fs::metadata(fixture.root.join("model/tiny.safetensors"))
            .unwrap()
            .len()
            - 8
    );
    for field in ["metadata_ready_seconds", "peak_rss_mib"] {
        let sample = value[field].as_f64().unwrap();
        assert!(
            sample.is_finite() && sample >= 0.0,
            "diagnostic sample assertion: {field}"
        );
    }
    assert!(
        value["resources"].is_object(),
        "resource observation object assertion"
    );
    assert_eq!(
        value["resources"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["configured", "effective", "scope"]
    );
    assert_eq!(
        value["resources"]["scope"],
        "standalone Rust process; hosted/inference/build policies are separate"
    );
    assert_eq!(value["resources"]["effective"]["affinity_cpu_count"], 1);
    assert_eq!(
        value["resources"]["effective"]["reserved_workspace_bytes"],
        0
    );
    assert_eq!(
        value["resources"]["effective"]["active_numeric_operations"],
        0
    );
}
