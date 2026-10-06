//! CLI defaults over two tiny sources; occupied-port refusals start no service.
use serde_json::{json, Value};
use std::{
    net::TcpListener,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

struct Fixture {
    root: PathBuf,
    a: PathBuf,
    b: PathBuf,
    output: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-intake-comparison-defaults-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        let a = root.join("a");
        let b = root.join("b");
        let output = root.join("output");
        for path in [&a, &b, &output] {
            std::fs::create_dir_all(path).unwrap();
        }
        let header = br#"{"first":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"second":{"dtype":"BF16","shape":[2,4],"data_offsets":[16,32]}}"#;
        for (path, words) in [
            (
                &a,
                [
                    0x3f80u16, 0xc000, 0x8000, 0x4080, 0x3f00, 0xbf00, 0x4040, 0xc080, 0x3f00,
                    0xbf00, 0x3f00, 0xbf00, 0x3f00, 0xbf00, 0x3f00, 0xbf00,
                ],
            ),
            (
                &b,
                [
                    0x3f00u16, 0xbf80, 0x8000, 0x4000, 0x3e80, 0xbe80, 0x3fc0, 0xc000, 0x3e80,
                    0xbe80, 0x3e80, 0xbe80, 0x3e80, 0xbe80, 0x3e80, 0xbe80,
                ],
            ),
        ] {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            for bits in words {
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            std::fs::write(path.join("tiny.safetensors"), bytes).unwrap();
        }
        Self { root, a, b, output }
    }

    fn command(&self, name: &str, options: &[&str]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"));
        command
            .current_dir(&self.output)
            .arg(name)
            .arg("--model")
            .arg(&self.a);
        if name.starts_with("compare-") {
            command.arg("--compare-model").arg(&self.b);
        }
        command.args(options);
        bounded_output(command)
    }

    fn success(&self, name: &str, options: &[&str]) -> Value {
        let result = self.command(name, options);
        assert_eq!(
            result.status.code(),
            Some(0),
            "intake/comparison success: {name}; stderr={}",
            String::from_utf8_lossy(&result.stderr)
        );
        if name == "calibrate" {
            let progress = String::from_utf8(result.stderr)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(progress.len(), 2);
            for (id, tensor) in ["first", "second"].into_iter().enumerate() {
                assert_eq!(progress[id]["calibrated"], id);
                assert_eq!(progress[id]["name"], tensor);
                assert_eq!(progress[id]["values"], 8);
                assert_eq!(progress[id].as_object().unwrap().len(), 4);
                assert!(progress[id]["seconds"].as_f64().unwrap().is_finite());
            }
        } else {
            assert_eq!(result.stderr, b"", "intake/comparison success stderr");
        }
        serde_json::from_slice(&result.stdout).unwrap()
    }

    fn refusal(&self, name: &str, options: &[&str], message: &str) {
        assert_refusal(self.command(name, options), message);
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn bounded_output(mut command: Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut completed = false;
    while Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            completed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if !completed {
        child.kill().unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(completed, "bounded native CLI completion after cleanup");
    assert!(output.stdout.len() < 64 * 1024);
    assert!(output.stderr.len() < 64 * 1024);
    output
}

fn assert_refusal(output: Output, message: &str) {
    assert_eq!(
        output.status.code(),
        Some(1),
        "intake/comparison refusal exit"
    );
    assert_eq!(output.stdout, b"", "intake/comparison refusal stdout");
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        format!("ERROR: {message}\n"),
        "intake/comparison exact refusal"
    );
}

fn occupied_port(port: u16) -> (Option<TcpListener>, String) {
    // Retain an owned reservation when free; an already occupied port exercises
    // the same bind refusal without contacting or altering its owner.
    let reservation = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => Some(listener),
        Err(error) => {
            assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
            None
        }
    };
    let error = TcpListener::bind(("127.0.0.1", port)).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
    (reservation, error.to_string())
}

#[test]
fn intake_default_cache_and_required_model_are_held() {
    let fixture = Fixture::new();
    let result = fixture.success("calibrate", &[]);
    assert_eq!(result["model"]["calibration_complete"], true);
    assert!(
        fixture.output.join("cache").is_dir(),
        "intake default cache location"
    );
    let mut missing = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"));
    missing.current_dir(&fixture.output).arg("calibrate");
    assert_refusal(bounded_output(missing), "--model DIRECTORY is required");
}

#[test]
fn intake_serve_defaults_hold_port_and_hash_opt_in_order() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.a.join("model-api.json"),
        json!({"siblings":[{"rfilename":"tiny.safetensors","lfs":{"sha256":"0".repeat(64)}}]})
            .to_string(),
    )
    .unwrap();
    let (_reservation, message) = occupied_port(8775);
    fixture.refusal("serve", &[], &message);
    fixture.refusal(
        "serve",
        &["--verify-sha", "true"],
        "Full shard SHA mismatch against local pinned metadata",
    );
    let selected = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = selected.local_addr().unwrap().port().to_string();
    let error = TcpListener::bind(selected.local_addr().unwrap()).unwrap_err();
    fixture.refusal(
        "serve",
        &["--port", &port, "--verify-sha", "false"],
        &error.to_string(),
    );
}

#[test]
fn dispatch_keeps_inspect_and_exact_unknown_command_refusal() {
    let fixture = Fixture::new();
    let result = fixture.success("inspect", &["--tensor", "0", "--row", "0", "--col", "1"]);
    assert_eq!(result["raw_exact"], "-2", "dispatch inspect selection");
    assert_eq!(result["raw_hex_le"], "00c0");
    fixture.refusal("unknown-command", &[], "Unknown command; use --help");
}

#[test]
fn comparison_cache_tensor_coordinates_and_required_options_are_held() {
    let fixture = Fixture::new();
    let model = fixture.success("compare-metadata", &[]);
    assert_eq!(model["compatibility"]["tensor_count"], 2);
    assert!(
        fixture.output.join("cache-comparison").is_dir(),
        "comparison default cache location"
    );
    let result = fixture.success("compare-inspect", &[]);
    assert_eq!(result["pair_id"], 0);
    assert_eq!(result["native_indices"], json!([0, 0]));
    assert_eq!(result["originals"]["a"]["raw_exact"], "1");
    assert_eq!(result["originals"]["b"]["raw_exact"], "0.5");
    let selected = fixture.success(
        "compare-inspect",
        &["--tensor", "1", "--row", "1", "--col", "3"],
    );
    assert_eq!(selected["pair_id"], 1);
    assert_eq!(selected["native_indices"], json!([1, 3]));
    assert_eq!(selected["originals"]["a"]["raw_exact"], "-0.5");
    assert_eq!(selected["originals"]["b"]["raw_exact"], "-0.25");
    fixture.refusal(
        "compare-calibrate",
        &[],
        "Comparison calibration requires an explicit --tensor ID",
    );
    fixture.refusal("compare-tile", &[], "--out PREFIX required");
}

#[test]
fn comparison_tile_defaults_hold_delta_linear_native_coordinates() {
    let fixture = Fixture::new();
    fixture.success("compare-calibrate", &["--tensor", "0"]);
    let result = fixture.success("compare-tile", &["--out", "comparison"]);
    assert_eq!(result["quantity"], "delta", "comparison default quantity");
    assert_eq!(result["mapping"], "linear", "comparison default mapping");
    assert_eq!(result["pair_id"], 0);
    assert_eq!(result["metrics"]["width"], 4);
    assert_eq!(result["metrics"]["height"], 2);
    let bytes = std::fs::read(fixture.output.join("comparison.f64le")).unwrap();
    let values = bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![-0.25, 0.5, 0.0, -1.0, -0.125, 0.125, -0.75, 1.0],
        "comparison default exact field"
    );
}

#[test]
fn comparison_serve_default_and_override_keep_bind_refusals() {
    let fixture = Fixture::new();
    let (_reservation, message) = occupied_port(8776);
    fixture.refusal("compare-serve", &[], &message);
    let selected = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = selected.local_addr().unwrap().port().to_string();
    let error = TcpListener::bind(selected.local_addr().unwrap()).unwrap_err();
    fixture.refusal("compare-serve", &["--port", &port], &error.to_string());
}

#[test]
fn dispatch_catalog_keeps_unknown_comparison_constructor_and_parse_order() {
    let fixture = Fixture::new();
    fixture.refusal("compare-unknown", &[], "Unknown comparison command");
    fixture.refusal(
        "compare-unknown",
        &["--tensor", "not-an-index"],
        "invalid digit found in string",
    );
    let cache = fixture.a.to_str().unwrap();
    fixture.refusal(
        "compare-unknown",
        &["--cache", cache, "--tensor", "not-an-index"],
        "Comparison cache must be outside both source directories",
    );
    let mut missing = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"));
    missing
        .current_dir(&fixture.output)
        .arg("compare-unknown")
        .arg("--model")
        .arg(&fixture.a)
        .args(["--tensor", "not-an-index"]);
    assert_refusal(
        bounded_output(missing),
        "--compare-model DIRECTORY is required for comparison",
    );
}

#[test]
fn dispatch_catalog_keeps_private_channel_and_case_refusals() {
    let fixture = Fixture::new();
    fixture.refusal("hosted-renderer", &[], "Private channel required");
    fixture.refusal(
        "hosted-renderer",
        &["--channel-fd", "not-a-descriptor"],
        "invalid digit found in string",
    );
    fixture.refusal("Inspect", &[], "Unknown command; use --help");
    fixture.refusal("Compare-inspect", &[], "Unknown command; use --help");
}

#[test]
fn dispatch_catalog_keeps_eager_tensor_parse_for_comparison_metadata_and_serve() {
    let fixture = Fixture::new();
    fixture.refusal(
        "compare-metadata",
        &["--tensor", "not-an-index"],
        "invalid digit found in string",
    );
    fixture.refusal(
        "compare-serve",
        &["--tensor", "not-an-index", "--port", "65536"],
        "invalid digit found in string",
    );
}
