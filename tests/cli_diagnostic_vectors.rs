//! CLI report envelopes over four BF16 values; volatile samples stay observational.
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
use weight_atlas_rust::source::Source;

struct Fixture {
    root: PathBuf,
    model: PathBuf,
    cache: PathBuf,
    bytes: Vec<u8>,
    cpu: usize,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-cli-diagnostics-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        let model = root.join("model");
        let cache = root.join("cache");
        std::fs::create_dir_all(&model).unwrap();
        let header = br#"{"tiny":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        for word in [0x3f80u16, 0xc000, 0x3f00, 0xbf00] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        std::fs::write(model.join("tiny.safetensors"), &bytes).unwrap();
        // SAFETY: initialized writable cpu_set_t with the matching byte size.
        let cpu = unsafe {
            let mut allowed: libc::cpu_set_t = std::mem::zeroed();
            assert_eq!(
                libc::sched_getaffinity(0, std::mem::size_of_val(&allowed), &mut allowed),
                0
            );
            (0..libc::CPU_SETSIZE as usize)
                .find(|&id| libc::CPU_ISSET(id, &allowed))
                .unwrap()
        };
        Self {
            root,
            model,
            cache,
            bytes,
            cpu,
        }
    }

    fn command(&self, name: &str, options: &[&str]) -> (Value, String) {
        let output = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
            .current_dir(&self.root)
            .env("ATLAS_CPU", self.cpu.to_string())
            .arg(name)
            .arg("--model")
            .arg(&self.model)
            .arg("--cache")
            .arg(&self.cache)
            .args(options)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(0),
            "diagnostic {name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.len() < 64 * 1024);
        (
            serde_json::from_slice(&output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    }

    fn calibrate(&self) -> Value {
        let (value, stderr) = self.command("calibrate", &[]);
        let progress: Value = serde_json::from_str(stderr.trim()).unwrap();
        keys(&progress, &["calibrated", "name", "values", "seconds"]);
        assert_eq!(progress["calibrated"], 0);
        assert_eq!(progress["name"], "tiny");
        assert_eq!(progress["values"], 4);
        sample(&progress, "seconds");
        value
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn keys(value: &Value, names: &[&str]) {
    let actual = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut expected = names.to_vec();
    expected.sort_unstable();
    assert_eq!(actual, expected, "exact diagnostic envelope");
}

fn sample(value: &Value, name: &str) {
    let number = value[name].as_f64().expect("numeric diagnostic sample");
    assert!(
        number.is_finite() && number >= 0.0,
        "nonnegative finite {name}"
    );
}

#[test]
fn calibration_report_envelope() {
    let fixture = Fixture::new();
    let value = fixture.calibrate();
    keys(
        &value,
        &[
            "model",
            "wall_seconds",
            "peak_rss_mib",
            "minimum_available_gib",
            "cpu",
        ],
    );
    assert_eq!(value["model"]["calibration_complete"], true);
    assert_eq!(value["cpu"], fixture.cpu);
    for field in ["wall_seconds", "peak_rss_mib", "minimum_available_gib"] {
        sample(&value, field);
    }
}

#[test]
fn tile_report_envelope() {
    let fixture = Fixture::new();
    fixture.calibrate();
    let (value, stderr) = fixture.command("tile", &[]);
    assert_eq!(stderr, "");
    keys(&value, &["metrics", "seconds", "peak_rss_mib"]);
    assert_eq!(value["metrics"]["source_count"], 4);
    assert_eq!(value["metrics"]["source_bytes_read"], 8);
    sample(&value, "seconds");
    sample(&value, "peak_rss_mib");
}

#[test]
fn benchmark_report_envelope() {
    let fixture = Fixture::new();
    fixture.calibrate();
    let (value, stderr) = fixture.command("bench", &["--repeats", "1"]);
    assert_eq!(stderr, "");
    keys(&value, &["records", "wall_seconds", "peak_rss_mib", "cpu"]);
    assert_eq!(value["cpu"], fixture.cpu);
    assert!(!value["records"].as_array().unwrap().is_empty());
    for record in value["records"].as_array().unwrap() {
        assert_eq!(record["tensor"], "tiny");
        assert_eq!(record["shape"], json!([2, 2]));
        assert_eq!(record["runs_seconds"].as_array().unwrap().len(), 1);
        assert!(record["last"]["png_bytes"].as_u64().unwrap() > 0);
    }
    sample(&value, "wall_seconds");
    sample(&value, "peak_rss_mib");
}

#[test]
fn metadata_report_envelope() {
    let fixture = Fixture::new();
    let source = Source::open(&fixture.model).unwrap();
    let (value, stderr) = fixture.command("metadata", &[]);
    assert_eq!(stderr, "");
    keys(
        &value,
        &[
            "source_directory",
            "source_identity",
            "header_bytes",
            "source_bytes",
            "tensor_count",
            "parameter_count",
            "catalog",
            "shards",
            "elapsed_seconds",
            "peak_rss_mib",
            "cpu",
            "full_sha_recomputed",
        ],
    );
    assert_eq!(value["source_directory"], json!(fixture.model));
    assert_eq!(value["source_identity"], source.identity);
    assert_eq!(value["header_bytes"], fixture.bytes.len() - 8);
    assert_eq!(value["source_bytes"], fixture.bytes.len());
    assert_eq!(value["tensor_count"], 1);
    assert_eq!(value["parameter_count"], 4);
    assert_eq!(value["catalog"], json!(source.tensors));
    assert_eq!(value["shards"], json!(source.shards));
    assert_eq!(value["full_sha_recomputed"], false);
    assert_eq!(value["cpu"], fixture.cpu);
    sample(&value, "elapsed_seconds");
    sample(&value, "peak_rss_mib");
}

#[test]
fn verification_report_envelope_and_saved_bytes() {
    let fixture = Fixture::new();
    let source = Source::open(&fixture.model).unwrap();
    let (value, stderr) = fixture.command("verify", &[]);
    assert_eq!(stderr, "Hashed tiny.safetensors\n");
    keys(
        &value,
        &[
            "source_identity",
            "shards",
            "wall_seconds",
            "peak_rss_mib",
            "minimum_available_gib",
            "cpu",
            "scope",
        ],
    );
    assert_eq!(value["source_identity"], source.identity);
    assert_eq!(
        value["shards"],
        json!([{"shard":"tiny.safetensors", "sha256":weight_atlas_rust::sha(&fixture.bytes), "matches_saved_expected_sha":null, "bytes":fixture.bytes.len()}])
    );
    assert_eq!(value["scope"], "Fresh full source SHA; comparisons use explicitly selected local metadata, not a new remote trust check");
    assert_eq!(value["cpu"], fixture.cpu);
    for field in ["wall_seconds", "peak_rss_mib", "minimum_available_gib"] {
        sample(&value, field);
    }
    assert_eq!(
        std::fs::read(fixture.cache.join("verification.json")).unwrap(),
        serde_json::to_vec_pretty(&value).unwrap()
    );
}
