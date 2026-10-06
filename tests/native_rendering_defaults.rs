//! Native CLI defaults over sixteen synthetic values, with no inference or service.
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture {
    root: PathBuf,
    model: PathBuf,
    cache: PathBuf,
    output: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-render-defaults-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        let model = root.join("model");
        let cache = root.join("cache");
        let output = root.join("output");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        let header = br#"{"first":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"second":{"dtype":"BF16","shape":[2,4],"data_offsets":[16,32]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        for bits in [
            0x3f80u16, 0xc000, 0x8000, 0x4080, 0x3f00, 0xbf00, 0x4040, 0xc080, 0x3f00, 0xbf00,
            0x3f00, 0xbf00, 0x3f00, 0xbf00, 0x3f00, 0xbf00,
        ] {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        std::fs::write(model.join("tiny.safetensors"), bytes).unwrap();
        let fixture = Self {
            root,
            model,
            cache,
            output,
        };
        let calibrated = fixture.command("calibrate", &[]);
        assert_eq!(
            calibrated.status.code(),
            Some(0),
            "synthetic calibration setup"
        );
        let progress = String::from_utf8(calibrated.stderr)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(progress.len(), 2, "synthetic calibration progress count");
        for (index, name) in ["first", "second"].into_iter().enumerate() {
            assert_eq!(progress[index]["calibrated"], index);
            assert_eq!(progress[index]["name"], name);
            assert_eq!(progress[index]["values"], 8);
            assert!(progress[index]["seconds"].as_f64().unwrap().is_finite());
            assert_eq!(progress[index].as_object().unwrap().len(), 4);
        }
        let report: Value = serde_json::from_slice(&calibrated.stdout).unwrap();
        assert_eq!(report["model"]["calibration_complete"], true);
        fixture
    }

    fn command(&self, command: &str, options: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
            .current_dir(&self.output)
            .arg(command)
            .arg("--model")
            .arg(&self.model)
            .arg("--cache")
            .arg(&self.cache)
            .args(options)
            .output()
            .unwrap()
    }

    fn success(&self, command: &str, options: &[&str]) -> Value {
        let output = self.command(command, options);
        assert_eq!(
            output.status.code(),
            Some(0),
            "render default command success: {command}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stderr, b"", "render default success stderr");
        assert!(output.stdout.len() < 64 * 1024);
        serde_json::from_slice(&output.stdout).unwrap()
    }

    fn refusal(&self, command: &str, options: &[&str], message: &str) {
        let output = self.command(command, options);
        assert_eq!(output.status.code(), Some(1), "render default refusal exit");
        assert_eq!(output.stdout, b"", "render default refusal stdout");
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            format!("ERROR: {message}\n"),
            "render default exact refusal"
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn fields(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

#[test]
fn rendering_default_overview_keeps_rules_budget_overrides_and_refusals() {
    let fixture = Fixture::new();
    let result = fixture.success("overview", &["--tensor", "0"]);
    assert_eq!(result["max_values"], 16777216, "overview default budget");
    assert_eq!(result["metrics"]["width"], 4);
    assert_eq!(result["metrics"]["height"], 2);
    let rules = result["tiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tile| tile["rule"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        rules,
        vec!["tensor_linear", "tensor_asinh"],
        "overview default rules/order"
    );
    let explicit = fixture.success(
        "overview",
        &[
            "--tensor",
            "1",
            "--slice",
            "",
            "--rules",
            "tensor_magnitude",
            "--max-values",
            "8",
        ],
    );
    assert_eq!(explicit["max_values"], 8);
    assert_eq!(explicit["tiles"].as_array().unwrap().len(), 1);
    assert_eq!(explicit["tiles"][0]["rule"], "tensor_magnitude");
    fixture.refusal("overview", &[], "Overview requires explicit --tensor ID");
    fixture.refusal(
        "overview",
        &["--tensor", "0", "--max-values", "0"],
        "Overview max-values must be 1–67108864",
    );
}

#[test]
fn rendering_default_tile_keeps_tensor_rules_level_coordinates_and_output() {
    let fixture = Fixture::new();
    let result = fixture.success("tile", &[]);
    assert_eq!(result["metrics"]["width"], 4);
    assert_eq!(result["metrics"]["height"], 2);
    let mut names = std::fs::read_dir(&fixture.output)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        vec![
            "tile-global_asinh.f64le",
            "tile-global_asinh.png",
            "tile-global_linear.f64le",
            "tile-global_linear.png",
        ],
        "tile default output names/rules"
    );
    assert_eq!(
        fields(&fixture.output.join("tile-global_linear.f64le")),
        vec![0.25, -0.5, 0.0, 1.0, 0.125, -0.125, 0.75, -1.0],
        "tile default tensor and coordinates"
    );
    fixture.success(
        "tile",
        &[
            "--tensor",
            "1",
            "--slice",
            "",
            "--rules",
            "tensor_linear",
            "--level",
            "2",
            "--x",
            "0",
            "--y",
            "0",
            "--out",
            "selected",
        ],
    );
    assert_eq!(
        fields(&fixture.output.join("selected-tensor_linear.f64le")),
        vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        "tile explicit overrides"
    );
}

#[test]
fn rendering_default_bench_keeps_tensor_repeats_and_factor_order() {
    let fixture = Fixture::new();
    let result = fixture.success("bench", &[]);
    let records = result["records"].as_array().unwrap();
    assert_eq!(
        records
            .iter()
            .map(|record| record["factor"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 4],
        "bench default factor order"
    );
    for record in records {
        assert_eq!(record["tensor"], "first", "bench default tensor");
        assert_eq!(record["shape"], serde_json::json!([2, 4]));
        assert_eq!(
            record["runs_seconds"].as_array().unwrap().len(),
            3,
            "bench default repeats"
        );
    }
    let explicit = fixture.success("bench", &["--tensor", "1", "--repeats", "2"]);
    for record in explicit["records"].as_array().unwrap() {
        assert_eq!(record["tensor"], "second");
        assert_eq!(record["runs_seconds"].as_array().unwrap().len(), 2);
    }
    fixture.refusal("bench", &["--repeats", "0"], "repeats 1–30");
}

#[test]
fn rendering_default_repeated_options_keep_the_early_exact_refusal() {
    let output = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
        .args(["bench", "--repeats", "2", "--repeats", "4"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout, b"");
    assert_eq!(output.stderr, b"ERROR: Duplicate CLI option\n");
}
