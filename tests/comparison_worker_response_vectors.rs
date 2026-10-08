//! Bounded comparison worker requests over two pairs of tiny BF16 tensors.
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
        assert_eq!(
            self.status, status,
            "comparison worker refusal status: {message}"
        );
        assert_eq!(
            self.json(),
            json!({"api_version":1,"error":message}),
            "comparison worker refusal body"
        );
    }
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-comparison-worker-vectors-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists(), "comparison worker fixture root is new");
        let model = root.join("a");
        let compare_model = root.join("b");
        let header = json!({
            "first":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]},
            "second":{"dtype":"BF16","shape":[1,2],"data_offsets":[4,8]}
        })
        .to_string();
        for (path, words) in [
            (&model, [0x3f80u16, 0xc000, 0x4040, 0x4080]),
            (&compare_model, [0x4040u16, 0xbf80, 0xbf80, 0x4000]),
        ] {
            std::fs::create_dir_all(path).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            for word in words {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            std::fs::write(path.join("tiny.safetensors"), bytes).unwrap();
        }
        let mut fixture = Self {
            root,
            child: None,
            address: None,
        };
        let child = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
            .current_dir(&fixture.root)
            .arg("compare-serve")
            .arg("--model")
            .arg(&model)
            .arg("--compare-model")
            .arg(&compare_model)
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
        eprintln!("comparison worker child pid: {}", child.id());
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
                "comparison worker server remains live during startup"
            );
            let text = std::fs::read_to_string(fixture.root.join("stdout")).unwrap();
            assert!(
                text.len() < 32 * 1024,
                "comparison worker startup output bound"
            );
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
        panic!("comparison worker startup deadline assertion");
    }

    fn request(&self, method: &str, path: &str) -> Response {
        let address = self.address.unwrap();
        let connected = TcpStream::connect_timeout(&address, Duration::from_secs(3));
        assert!(
            connected.is_ok(),
            "comparison worker connection assertion: {connected:?}"
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
        assert!(sent.is_ok(), "comparison worker write assertion: {sent:?}");
        let mut bytes = Vec::new();
        let received = socket.take(64 * 1024 + 1).read_to_end(&mut bytes);
        assert!(
            received.is_ok(),
            "comparison worker response read assertion: {received:?}"
        );
        assert!(
            bytes.len() <= 64 * 1024,
            "comparison worker response bound assertion"
        );
        let end = bytes.windows(4).position(|window| window == b"\r\n\r\n");
        assert!(
            end.is_some(),
            "comparison worker response framing assertion"
        );
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
            "comparison worker complete response length"
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
            let response = self.request("GET", "/api/comparison/model");
            assert_eq!(
                response.status, 200,
                "comparison worker status polling response"
            );
            last = response.json();
            if complete(&last) {
                return last;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("comparison worker calibration completion assertion: {last}");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let reaped = child.wait().is_ok();
            eprintln!("comparison worker lifecycle: reaped={reaped}");
            for name in ["stdout", "stderr"] {
                let text = std::fs::read_to_string(self.root.join(name)).unwrap_or_default();
                eprintln!("comparison worker {name}: {text}");
            }
            assert!(reaped, "comparison worker child reap assertion");
        }
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn comparison_worker_preserves_status_defaults_calibration_and_provenance() {
    let fixture = Fixture::new();
    let model = fixture.request("GET", "/api/comparison/model");
    assert_eq!(model.status, 200, "comparison worker model response");
    let identity = model.json();
    fixture
        .request("GET", "/api/comparison/tile")
        .refusal(503, "Complete paired-tensor calibration is not ready");
    fixture
        .request("GET", "/api/comparison/tile?tensor=bad")
        .refusal(400, "invalid digit found in string");
    fixture
        .request("GET", "/api/comparison/tile?quantity=unknown")
        .refusal(400, "Unsupported comparison quantity");
    fixture
        .request("GET", "/api/comparison/tile?mapping=unknown")
        .refusal(
            400,
            "Unsupported comparison mapping; exact paired robust/percentile calibration is pending",
        );
    let queued = fixture.request("POST", "/api/comparison/calibrate?tensor=0");
    assert_eq!(queued.status, 202, "comparison worker calibration accepted");
    assert_eq!(
        queued.json()["queued"],
        0,
        "comparison worker queued identity"
    );
    let complete =
        fixture.wait_status(|status| status["catalog"][0]["calibration_complete"] == true);
    assert_eq!(
        complete["catalog"][0]["scales"],
        json!({"count":2,"shared_raw_max":3.0,"difference_max":2.0,"nonzero_difference_count":2}),
        "comparison worker exact first pair scales"
    );
    assert_eq!(
        complete["catalog"][1]["calibration_complete"], false,
        "comparison worker selected pair only"
    );
    let tile = fixture.request("GET", "/api/comparison/tile");
    assert_eq!(tile.status, 200, "comparison worker default tile success");
    assert_eq!(
        tile.headers["content-type"], "image/png",
        "comparison worker PNG MIME"
    );
    assert_eq!(
        tile.headers["x-atlas-factor"], "2",
        "comparison worker default level factor"
    );
    assert_eq!(
        tile.headers["x-atlas-source-bytes"], "8",
        "comparison worker both tiny source byte counts"
    );
    assert_eq!(
        tile.headers["x-atlas-cache"], "miss",
        "comparison worker first tile miss"
    );
    assert_eq!(
        tile.headers["x-atlas-coordinate-space"], "checkpoint-comparison-v1",
        "comparison worker coordinate space"
    );
    assert_eq!(
        tile.headers["x-atlas-inference-editable"], "false",
        "comparison worker editability"
    );
    assert_eq!(
        tile.headers["x-atlas-comparison-identity"],
        identity["comparison_identity"].as_str().unwrap(),
        "comparison worker comparison identity"
    );
    assert_eq!(
        tile.headers["x-atlas-source-a-identity"],
        identity["sources"]["a"]["source_identity"]
            .as_str()
            .unwrap(),
        "comparison worker source A identity"
    );
    assert_eq!(
        tile.headers["x-atlas-source-b-identity"],
        identity["sources"]["b"]["source_identity"]
            .as_str()
            .unwrap(),
        "comparison worker source B identity"
    );
    assert_ne!(
        tile.headers["x-atlas-source-a-identity"], tile.headers["x-atlas-source-b-identity"],
        "comparison worker distinct source identities"
    );
    assert!(
        tile.body.starts_with(b"\x89PNG\r\n\x1a\n"),
        "comparison worker PNG framing assertion"
    );
    let explicit = fixture.request(
        "GET",
        "/api/comparison/tile?tensor=0&quantity=delta&mapping=linear&level=0&x=0&y=0",
    );
    assert_eq!(
        explicit.status, 200,
        "comparison worker explicit defaults success"
    );
    assert_eq!(
        explicit.body, tile.body,
        "comparison worker default and explicit PNG bytes"
    );
    assert_eq!(
        explicit.headers["x-atlas-cache"], "hit",
        "comparison worker shared default cache key"
    );
    assert_eq!(
        explicit.headers["x-atlas-source-bytes"], "0",
        "comparison worker cached source bytes"
    );
    fixture
        .request("GET", "/api/comparison/tile?x=1")
        .refusal(400, "Comparison tile outside native shape");
    fixture
        .request("GET", "/api/comparison/tile?tensor=1")
        .refusal(503, "Complete paired-tensor calibration is not ready");
    let queued_second = fixture.request("POST", "/api/comparison/calibrate?tensor=1");
    assert_eq!(
        queued_second.status, 202,
        "comparison worker second pair accepted"
    );
    let all = fixture.wait_status(|status| status["catalog"][1]["calibration_complete"] == true);
    assert_eq!(
        all["catalog"][1]["scales"],
        json!({"count":2,"shared_raw_max":4.0,"difference_max":4.0,"nonzero_difference_count":2}),
        "comparison worker exact second pair scales"
    );
    assert_eq!(
        fixture
            .request("GET", "/api/comparison/tile?tensor=1")
            .status,
        200,
        "comparison worker second pair tile ready"
    );
}

#[test]
fn startup_diagnostic_envelope_keeps_comparison_coordinates() {
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
            "comparison_identity",
            "coordinate_space",
            "inference_editable",
            "listening",
            "peak_rss_mib",
            "tensors"
        ]
    );
    assert_eq!(
        value["listening"],
        format!("http://{}", fixture.address.unwrap())
    );
    assert_eq!(value["tensors"], 2);
    assert_eq!(value["coordinate_space"], "checkpoint-comparison-v1");
    assert_eq!(value["inference_editable"], false);
    assert_eq!(
        value["comparison_identity"],
        fixture.request("GET", "/api/comparison/model").json()["comparison_identity"]
    );
    let peak = value["peak_rss_mib"].as_f64().unwrap();
    assert!(
        peak.is_finite() && peak >= 0.0,
        "comparison peak observation assertion"
    );
}
