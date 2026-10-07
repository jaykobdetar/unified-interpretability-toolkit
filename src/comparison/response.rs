//! Private presentation records for the existing comparison JSON envelopes.
use serde_json::{json, Value};
use std::path::Path;

pub(super) struct Identity<'a> {
    pub version: &'static str,
    pub comparison_identity: &'a str,
    pub source_a_identity: &'a str,
    pub source_a_directory: &'a Path,
    pub source_b_identity: &'a str,
    pub source_b_directory: &'a Path,
}

pub(super) struct Model<'a> {
    pub identity: Value,
    pub catalog: Vec<Value>,
    pub tensor_count: usize,
    pub calibration_note: &'a str,
}

pub(super) struct View {
    pub identity: Value,
    pub pair: Value,
    pub legends: Value,
}

pub(super) struct Inspection<'a> {
    pub identity: Value,
    pub pair_id: usize,
    pub name: &'a str,
    pub row: usize,
    pub col: usize,
    pub native_indices: Vec<usize>,
    pub original_a: Value,
    pub original_b: Value,
    pub delta: Option<f64>,
    pub finite: bool,
}

impl From<Identity<'_>> for Value {
    fn from(identity: Identity<'_>) -> Self {
        json!({
            "api_version":1,
            "extension":identity.version,
            "comparison_identity":identity.comparison_identity,
            "coordinate_space":identity.version,
            "inference_editable":false,
            "sources":{"a":{"source_identity":identity.source_a_identity,"source_directory":identity.source_a_directory},"b":{"source_identity":identity.source_b_identity,"source_directory":identity.source_b_directory}},
            "identity_validation":"Both canonical paths, complete headers/index and shard fingerprints; not fresh full-content hashes",
            "full_sha_recomputed":false
        })
    }
}

impl From<Model<'_>> for Value {
    fn from(model: Model<'_>) -> Self {
        let mut v = model.identity;
        v["catalog"] = json!(model.catalog);
        v["compatibility"] = json!({
            "complete":true,
            "policy":"identical complete named tensor sets and native shapes; BF16/F16/F32 mixed pairs permitted",
            "tensor_count":model.tensor_count
        });
        v["mappings"] = json!(["linear", "asinh", "magnitude"]);
        v["unsupported_mappings"] = json!({
            "robust99":"Exact paired-distribution calibration pending",
            "signed_percentile":"Exact paired rank index pending",
            "global":"Comparison scales are tensor-scoped"
        });
        v["quantities"] = json!(["a", "b", "delta", "abs_delta"]);
        v["calibration_note"] = json!(model.calibration_note);
        v
    }
}

impl From<View> for Value {
    fn from(view: View) -> Self {
        let mut v = view.identity;
        v["pair"] = view.pair;
        v["tile_size"] = json!(256);
        v["legends"] = view.legends;
        v
    }
}

impl From<Inspection<'_>> for Value {
    fn from(inspection: Inspection<'_>) -> Self {
        let mut v = inspection.identity;
        v["pair_id"] = json!(inspection.pair_id);
        v["name"] = json!(inspection.name);
        v["row"] = json!(inspection.row);
        v["col"] = json!(inspection.col);
        v["native_indices"] = json!(inspection.native_indices);
        v["originals"] = json!({
            "a":inspection.original_a,
            "b":inspection.original_b
        });
        v["difference"] = json!({
            "value":inspection.delta,
            "decimal_f64":inspection.delta.map(|d|d.to_string()),
            "direction":"B-A",
            "arithmetic":"F64 subtraction of exactly decoded originals; not exact symbolic subtraction",
            "derived":true,
            "original_source_value":false,
            "unavailable_reason":if inspection.finite{None}else{Some("Nonfinite original; difference unavailable")}
        });
        v
    }
}
