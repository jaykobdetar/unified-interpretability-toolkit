//! Model presentation vectors from tiny headers and explicit in-memory state.
#[path = "support/workspace.rs"]
mod workspace;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::UNIX_EPOCH,
};
use weight_atlas_rust::{render::Stats, state::State};

struct Fixture {
    root: PathBuf,
    state: State,
    header_len: usize,
    mixed: bool,
}

impl Fixture {
    fn new(mixed: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-model-vectors-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let model = root.join("model-λ");
        std::fs::create_dir_all(&model).unwrap();
        let mut header = json!({
            "b-vector":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},
            "c-volume":{"dtype":"F16","shape":[2,1,2],"data_offsets":[4,12]},
            "d-single":{"dtype":"F32","shape":[1,2],"data_offsets":[12,20]}
        });
        let payload_len = if mixed {
            header["a-scalar"] = json!({"dtype":"BF16","shape":[],"data_offsets":[20,22]});
            header["e-integer"] = json!({"dtype":"I8","shape":[3],"data_offsets":[22,25]});
            header["f-scalar32"] = json!({"dtype":"F32","shape":[],"data_offsets":[25,29]});
            header["g-empty"] = json!({"dtype":"BF16","shape":[0],"data_offsets":[29,29]});
            29
        } else {
            20
        };
        let header = header.to_string();
        let header_len = header.len();
        let mut file = (header_len as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.resize(file.len() + payload_len, 0);
        std::fs::write(model.join("tiny.safetensors"), file).unwrap();
        let mut state = State::open(
            &model,
            &root.join("cache"),
            Some("Model \"λ\"".into()),
            Some("revision/α".into()),
        )
        .unwrap();
        state.calibration_note = "held note: λ\nsecond line".into();
        Self {
            root,
            state,
            header_len,
            mixed,
        }
    }

    fn calibrations(&self) {
        let mut calibration = self.state.calibration.lock().unwrap();
        let first = usize::from(self.mixed);
        for (offset, (count, dtype, maximum)) in
            [(2, "BF16", 3.0), (4, "F16", 4.0), (2, "F32", 9.5)]
                .into_iter()
                .enumerate()
        {
            calibration.tensors.insert(
                first + offset,
                Stats {
                    count,
                    max_abs: maximum,
                    median_nonzero_abs: 0.5,
                    q99: 1.75,
                    robust_clipped_count: 1,
                    histogram_sha256: Some("unused presentation fixture".into()),
                    exact_zero_count: 2,
                    unique_bit_patterns: if dtype == "F32" { None } else { Some(3) },
                    dtype: dtype.into(),
                    calibration_method: format!("exact {dtype} λ"),
                    seconds: 123.0,
                },
            );
        }
    }

    fn catalog(&self, calibrated: bool) -> Value {
        // These explicit descriptors are independent of Source's returned catalog.
        let mut rows = vec![
            (
                "b-vector",
                "BF16",
                vec![2],
                1,
                2,
                2,
                2,
                0,
                1,
                None,
                vec![0],
                false,
                3.0,
            ),
            (
                "c-volume",
                "F16",
                vec![2, 1, 2],
                1,
                2,
                4,
                2,
                4,
                1,
                None,
                vec![1, 2],
                true,
                4.0,
            ),
            (
                "d-single",
                "F32",
                vec![1, 2],
                1,
                2,
                2,
                4,
                12,
                1,
                None,
                vec![0, 1],
                false,
                9.5,
            ),
        ];
        if self.mixed {
            rows.insert(
                0,
                (
                    "a-scalar",
                    "BF16",
                    vec![],
                    0,
                    0,
                    1,
                    2,
                    20,
                    0,
                    Some("Scalar/empty tensor display unsupported"),
                    vec![],
                    false,
                    0.0,
                ),
            );
            rows.extend([
                ("e-integer", "I8", vec![3], 1, 3, 3, 1, 22, 2, Some("Storage extent validated; numeric encoding/quantization scale semantics unsupported"), vec![0], false, 0.0),
                ("f-scalar32", "F32", vec![], 0, 0, 1, 4, 25, 0, Some("Scalar/empty tensor display unsupported"), vec![], false, 0.0),
                ("g-empty", "BF16", vec![0], 1, 0, 0, 2, 29, 0, Some("Scalar/empty tensor display unsupported"), vec![0], false, 0.0),
            ]);
        }
        Value::Array(rows.into_iter().enumerate().map(|(id, (name, dtype, shape, rows, cols, count, width, offset, level, unavailable, axes, slice, maximum))| {
            let ready = calibrated && unavailable.is_none();
            let rule = if unavailable.is_some() { "unavailable: numeric tensor viewing unsupported" }
                else if dtype == "F32" { "unsupported: exact F32 absolute-value rank index pending" }
                else if ready { "supported dtype; requires a valid exact histogram" }
                else { "calibration pending" };
            let mut result = json!({
                "id":id,"name":name,"shape":shape,"rows":rows,"cols":cols,"count":count,
                "dtype":dtype,"element_bytes":width,"available":unavailable.is_none(),
                "unavailable_reason":unavailable,"shard":"tiny.safetensors","shard_id":0,
                "byte_offset":self.header_len+8+offset,"max_level":level,"min_level":0,
                "calibration_complete":ready,"slice_required":slice,"display_axes":axes,
                "max_abs":null,"median_nonzero_abs":null,"q99":null,
                "quantile_order_statistics":null,"quantile_interpolation":null,
                "robust_clipped_count":null,"exact_zero_count":null,"unique_bit_patterns":null,
                "calibration_method":null,"rule_status":{"tensor_signed_percentile":rule}
            });
            if ready {
                for (key, value) in [
                    ("max_abs",json!(maximum)), ("median_nonzero_abs",json!(0.5)), ("q99",json!(1.75)),
                    ("quantile_order_statistics",json!("exact")),
                    ("quantile_interpolation",json!("linear in F64; final floating-point rounding possible")),
                    ("robust_clipped_count",json!(1)), ("exact_zero_count",json!(2)),
                    ("unique_bit_patterns",if dtype == "F32" { Value::Null } else { json!(3) }),
                    ("calibration_method",json!(format!("exact {dtype} λ"))),
                ] { result[key] = value; }
            }
            result
        }).collect())
    }

    fn expected(&self, calibrated: bool, busy: bool) -> Value {
        let complete = calibrated && !self.mixed;
        let identity = format!(
            "{:x}",
            Sha256::digest(
                json!([
                    "weight-atlas-model-v1",
                    self.state.source.identity,
                    "revision/α"
                ])
                .to_string()
                .as_bytes()
            )
        );
        let rules: Value =
            serde_json::from_str(include_str!("fixtures/rule_metadata_vectors.json")).unwrap();
        let mut result = json!({
            "api_version":1,
            "extensions":["progressive-calibration-v1","source-dtypes-v1","trailing-slices-v1","source-binding-v2"],
            "backend":"Rust","name":"Model \"λ\"","revision":"revision/α",
            "representation":"Original BF16/F16/F32 source values; no model execution",
            "parameter_count":if self.mixed {13} else {8},"global_max":if complete {json!(9.5)} else {Value::Null},
            "calibration_complete":complete,"source_directory":self.root.join("model-λ"),
            "source_bytes":self.header_len+8+if self.mixed {29} else {20},"header_bytes_read":self.header_len+8,
            "source_identity":self.state.source.identity,"model_identity":identity,
            "identity_validation":"held note: λ\nsecond line","fresh_source_hashes":null,
            "catalog":self.catalog(calibrated),"rules":rules,
            "global_calibration_supported":!self.mixed,
            "global_calibration_unavailable_reason":if self.mixed {json!("Catalog includes unavailable tensors; global calibration cannot complete. Supported tensors remain eligible for local calibration.")} else {Value::Null},
            "supported_calibration_complete":calibrated,
            "render_semantics":"Each pooled pixel is the F64 mean of the pointwise transformed field over aligned source blocks. Only real edge addresses count. Summation/libm rounding can differ from Python; raw source values are unchanged."
        });
        result["coverage"] = json!({
            "source_complete":true,"sha_hashed_shards":if busy {5} else {0},
            "sha_expected_matched_shards":if busy {2} else {0},"sha_missing_expected_shards":if busy {2} else {0},
            "sha_verified_shards":if busy {2} else {0},
            "source_validation":"Complete header/index coverage; file identity checked. Freshly hashed shards and matches against saved local expectations are counted separately; saved expectations are not newly authenticated upstream.",
            "statistics_complete":complete,"values_streamed":if calibrated {8} else {0},
            "calibrated_tensors":if calibrated {3} else {0},"active_tensor":if busy {json!(1)} else {Value::Null},
            "calibration_error":if busy {json!("calibration detail: λ\nnot ready")} else {Value::Null},
            "all_requested":busy,"materialized_tiles":if busy {2} else {0},"materialized_bytes":if busy {37} else {0},
            "cache_budget_bytes":2147483648_u64,"fine_tile_file_cap":1000,"all_pixels_materialized":false,
            "rendering_policy":"On-demand numeric tiles. No full-model pixel pyramid required.",
            "supported_tensors":3,"unavailable_tensors":if self.mixed {4} else {0}
        });
        if busy {
            result["fresh_source_hashes"] = hash_records();
        }
        result
    }

    fn busy(&self) {
        let mut progress = self.state.progress.lock().unwrap();
        progress.active = Some(1);
        progress.all_requested = true;
        progress.error = Some("calibration detail: λ\nnot ready".into());
        let mut cache = self.state.cache.lock().unwrap();
        cache.entries.insert("first".into(), (12, UNIX_EPOCH));
        cache.entries.insert("second".into(), (25, UNIX_EPOCH));
        cache.bytes = 37;
        *self.state.fresh_hashes.lock().unwrap() = Some(hash_records());
    }

    fn check(&self, calibrated: bool, busy: bool) {
        assert_eq!(
            self.state.model().unwrap().to_string(),
            self.expected(calibrated, busy).to_string(),
            "model response bytes"
        );
    }
}

fn hash_records() -> Value {
    json!({"shards":[{"matches_saved_expected_sha":true},{"matches_saved_expected_sha":true},{"matches_saved_expected_sha":false},{"matches_saved_expected_sha":null},{}],"note":"held λ"})
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn model_mixed_catalog_pending_and_calibrated_bytes() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new(true);
    fixture.check(false, false);
    fixture.calibrations();
    fixture.check(true, false);
    fixture.busy();
    fixture.check(true, true);
}

#[test]
fn model_global_completion_hashes_progress_and_cache_bytes() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new(false);
    fixture.check(false, false);
    fixture.calibrations();
    fixture.busy();
    fixture.check(true, true);
    for hashes in [json!({}), json!({"shards":null}), json!({"shards":[]})] {
        *fixture.state.fresh_hashes.lock().unwrap() = Some(hashes.clone());
        let mut expected = fixture.expected(true, true);
        expected["fresh_source_hashes"] = hashes;
        for key in [
            "sha_hashed_shards",
            "sha_expected_matched_shards",
            "sha_missing_expected_shards",
            "sha_verified_shards",
        ] {
            expected["coverage"][key] = json!(0);
        }
        assert_eq!(
            fixture.state.model().unwrap().to_string(),
            expected.to_string(),
            "model missing or empty shard list"
        );
    }
}

#[test]
fn model_source_identity_refusal_precedes_response_construction() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new(false);
    std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.root.join("model-λ/tiny.safetensors"))
        .unwrap()
        .set_len(0)
        .unwrap();
    assert_eq!(
        fixture.state.model().unwrap_err().to_string(),
        "Source identity changed; stop and reopen model. Cached calibration/tiles are invalid."
    );
}
