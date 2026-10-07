//! Complete comparison presentation vectors over two tiny original sources.
#[path = "support/workspace.rs"]
mod workspace;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use weight_atlas_rust::comparison::Comparison;

struct Fixture {
    root: PathBuf,
    comparison: Comparison,
    offsets: [usize; 2],
    shape: Vec<usize>,
}

impl Fixture {
    fn new() -> Self {
        Self::shaped(&[2], false)
    }

    fn shaped(shape: &[usize], nonfinite: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-comparison-vectors-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut offsets = [0; 2];
        for (index, (label, dtype, raw)) in [
            (
                "a",
                "BF16",
                if nonfinite {
                    vec![0, 0, 0xc1, 0x7f]
                } else {
                    vec![0x80, 0xbf, 0x80, 0x3f]
                },
            ),
            (
                "b",
                "F32",
                if nonfinite {
                    vec![0, 0, 0, 0x80, 0, 0, 0x80, 0x7f]
                } else {
                    vec![0, 0, 0x40, 0x40, 0, 0, 0x80, 0xbf]
                },
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let path = root.join(label);
            std::fs::create_dir_all(&path).unwrap();
            let header = json!({"w.λ":{"dtype":dtype,"shape":shape,"data_offsets":[0,raw.len()]}})
                .to_string();
            offsets[index] = header.len() + 8;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(raw);
            std::fs::write(path.join("tiny.safetensors"), bytes).unwrap();
        }
        let mut comparison =
            Comparison::open(&root.join("a"), &root.join("b"), &root.join("cache")).unwrap();
        comparison.calibration_note = "held comparison note: λ\nsecond line".into();
        Self {
            root,
            comparison,
            offsets,
            shape: shape.to_vec(),
        }
    }

    fn identity(&self) -> Value {
        let source_a = &self.comparison.a.identity;
        let source_b = &self.comparison.b.identity;
        let identity = format!(
            "{:x}",
            Sha256::digest(
                json!({
                    "schema":"checkpoint-comparison-v1","a":source_a,"b":source_b,
                    "pairs":[pair(&self.shape)],"direction":"B-A"
                })
                .to_string()
                .as_bytes()
            )
        );
        json!({
            "api_version":1,"extension":"checkpoint-comparison-v1",
            "comparison_identity":identity,"coordinate_space":"checkpoint-comparison-v1",
            "inference_editable":false,
            "sources":{
                "a":{"source_identity":source_a,"source_directory":self.root.join("a")},
                "b":{"source_identity":source_b,"source_directory":self.root.join("b")}
            },
            "identity_validation":"Both canonical paths, complete headers/index and shard fingerprints; not fresh full-content hashes",
            "full_sha_recomputed":false
        })
    }

    fn model(&self, calibrated: bool, busy: bool) -> Value {
        let mut expected = self.identity();
        let mut tensor = pair(&self.shape);
        tensor["calibration_complete"] = json!(calibrated);
        tensor["scales"] = if calibrated {
            json!({"count":2,"shared_raw_max":3.0,"difference_max":4.0,"nonzero_difference_count":2})
        } else {
            Value::Null
        };
        expected["catalog"] = json!([tensor]);
        expected["compatibility"] = json!({"complete":true,"policy":"identical complete named tensor sets and native shapes; BF16/F16/F32 mixed pairs permitted","tensor_count":1});
        expected["mappings"] = json!(["linear", "asinh", "magnitude"]);
        expected["unsupported_mappings"] = json!({
            "robust99":"Exact paired-distribution calibration pending",
            "signed_percentile":"Exact paired rank index pending",
            "global":"Comparison scales are tensor-scoped"
        });
        expected["quantities"] = json!(["a", "b", "delta", "abs_delta"]);
        expected["calibration_note"] = json!("held comparison note: λ\nsecond line");
        expected["progress"] = if busy {
            json!({"active":0,"error":"held progress: λ\nnot ready"})
        } else {
            json!({"active":null,"error":null})
        };
        expected
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn pair(shape: &[usize]) -> Value {
    json!({"id":0,"name":"w.λ","shape":shape,"rows":1,"cols":2,"count":2,
        "max_level":1,"dtype_a":"BF16","dtype_b":"F32","a":0,"b":0})
}

fn legend(derived: bool) -> Value {
    json!({
        "quantity":if derived {"delta"} else {"a"},"mapping":"asinh",
        "original_source_values":!derived,"derived":derived,"difference_direction":"B-A",
        "min":if derived {-4.0} else {-3.0},"max":if derived {4.0} else {3.0},
        "bound":if derived {4.0} else {3.0},"s":if derived {0.04} else {0.03},
        "scope":if derived {"complete paired tensor; shared derived B-A scale"} else {"complete paired tensor; shared original A+B scale"},
        "calibration_domain":if derived {"difference"} else {"shared_raw"},"palette":"signed-blue-red",
        "formula":"asinh(v/s)/asinh(bound/s); s=bound/100; bound=0 => 0",
        "value_definition":if derived {"derived F64 arithmetic B-A; not an original weight"} else {"original source A weight"},
        "units":"native: selected quantity on labeled scale; pooled: mean of pointwise transformed quantity, not transform of a pooled weight",
        "rounding":"A/B decode exactly to F64; derived subtraction and field arithmetic may round"
    })
}

#[test]
fn comparison_identity_and_model_pending_ready_progress_bytes() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let comparison = &fixture.comparison;
    assert_eq!(
        comparison.identity_metadata().to_string(),
        fixture.identity().to_string(),
        "comparison identity bytes"
    );
    assert_eq!(
        comparison.model().unwrap().to_string(),
        fixture.model(false, false).to_string(),
        "comparison pending model bytes"
    );
    comparison.calibrate_one(0).unwrap();
    assert_eq!(
        comparison.model().unwrap().to_string(),
        fixture.model(true, false).to_string(),
        "comparison calibrated model bytes"
    );
    {
        let mut progress = comparison.progress.lock().unwrap();
        progress.active = Some(0);
        progress.error = Some("held progress: λ\nnot ready".into());
    }
    assert_eq!(
        comparison.model().unwrap().to_string(),
        fixture.model(true, true).to_string(),
        "comparison progress bytes"
    );
}

#[test]
fn comparison_view_envelope_legends_and_existing_refusals() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let comparison = &fixture.comparison;
    assert_eq!(
        comparison
            .view(0, "delta", "a", "asinh")
            .unwrap_err()
            .to_string(),
        "Complete paired-tensor calibration is not ready"
    );
    assert_eq!(
        comparison
            .view(1, "delta", "a", "asinh")
            .unwrap_err()
            .to_string(),
        "Unknown comparison pair"
    );
    comparison.calibrate_one(0).unwrap();
    let mut expected = fixture.identity();
    expected["pair"] = pair(&fixture.shape);
    expected["tile_size"] = json!(256);
    expected["legends"] = json!({"left":legend(true),"right":legend(false)});
    assert_eq!(
        comparison
            .view(0, "delta", "a", "asinh")
            .unwrap()
            .to_string(),
        expected.to_string(),
        "comparison view bytes"
    );
}

#[test]
fn comparison_inspection_originals_and_derived_direction_bytes() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    for (col, a, b, a_hex, b_hex, delta, decimal) in [
        (0, "-1", "3", "80bf", "00004040", 4.0, "4"),
        (1, "1", "-1", "803f", "000080bf", -2.0, "-2"),
    ] {
        let mut expected = fixture.identity();
        expected["pair_id"] = json!(0);
        expected["name"] = json!("w.λ");
        expected["row"] = json!(0);
        expected["col"] = json!(col);
        expected["native_indices"] = json!([col]);
        expected["originals"] = json!({
            "a":{"raw_exact":a,"raw_hex_le":a_hex,"dtype":"BF16","element_bytes":2,"shard":"tiny.safetensors","byte_offset":fixture.offsets[0]+col*2,"classification":"finite","original_source_value":true},
            "b":{"raw_exact":b,"raw_hex_le":b_hex,"dtype":"F32","element_bytes":4,"shard":"tiny.safetensors","byte_offset":fixture.offsets[1]+col*4,"classification":"finite","original_source_value":true}
        });
        expected["difference"] = json!({"value":delta,"decimal_f64":decimal,"direction":"B-A",
            "arithmetic":"F64 subtraction of exactly decoded originals; not exact symbolic subtraction",
            "derived":true,"original_source_value":false,"unavailable_reason":null});
        assert_eq!(
            fixture.comparison.inspect(0, 0, col).unwrap().to_string(),
            expected.to_string(),
            "comparison inspection bytes at column {col}"
        );
    }
}

#[test]
fn comparison_source_refusals_leave_identity_metadata_available() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let expected_identity = fixture.identity().to_string();
    std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.root.join("a/tiny.safetensors"))
        .unwrap()
        .set_len(0)
        .unwrap();
    let reason =
        "Source identity changed; stop and reopen model. Cached calibration/tiles are invalid.";
    assert_eq!(fixture.comparison.model().unwrap_err().to_string(), reason);
    assert_eq!(
        fixture
            .comparison
            .view(99, "unknown", "unknown", "unknown")
            .unwrap_err()
            .to_string(),
        reason
    );
    assert_eq!(
        fixture
            .comparison
            .inspect(99, 0, 0)
            .unwrap_err()
            .to_string(),
        reason
    );
    assert_eq!(
        fixture.comparison.identity_metadata().to_string(),
        expected_identity
    );
}

#[test]
fn comparison_matrix_indices_nonfinite_and_signed_zero_bytes() {
    let _isolation = workspace::guard();
    let fixture = Fixture::shaped(&[1, 2], true);
    for (col, a, b, a_hex, b_hex, a_class, b_class, value, decimal, unavailable) in [
        (
            0,
            "0.0",
            "-0.0",
            "0000",
            "00000080",
            "finite",
            "finite",
            json!(-0.0),
            json!("-0"),
            Value::Null,
        ),
        (
            1,
            "NaN",
            "Infinity",
            "c17f",
            "0000807f",
            "nan",
            "infinity",
            Value::Null,
            Value::Null,
            json!("Nonfinite original; difference unavailable"),
        ),
    ] {
        let mut expected = fixture.identity();
        expected["pair_id"] = json!(0);
        expected["name"] = json!("w.λ");
        expected["row"] = json!(0);
        expected["col"] = json!(col);
        expected["native_indices"] = json!([0, col]);
        expected["originals"] = json!({
            "a":{"raw_exact":a,"raw_hex_le":a_hex,"dtype":"BF16","element_bytes":2,"shard":"tiny.safetensors","byte_offset":fixture.offsets[0]+col*2,"classification":a_class,"original_source_value":true},
            "b":{"raw_exact":b,"raw_hex_le":b_hex,"dtype":"F32","element_bytes":4,"shard":"tiny.safetensors","byte_offset":fixture.offsets[1]+col*4,"classification":b_class,"original_source_value":true}
        });
        expected["difference"] = json!({"value":value,"decimal_f64":decimal,"direction":"B-A",
            "arithmetic":"F64 subtraction of exactly decoded originals; not exact symbolic subtraction",
            "derived":true,"original_source_value":false,"unavailable_reason":unavailable});
        assert_eq!(
            fixture.comparison.inspect(0, 0, col).unwrap().to_string(),
            expected.to_string(),
            "comparison matrix inspection bytes at column {col}"
        );
    }
}
