//! Typed readiness records at the existing JSON value boundary.
use serde_json::{Map, Value};

pub(super) struct TensorStatus<'a> {
    pub id: usize,
    pub calibration_complete: bool,
    pub max_abs: Option<f64>,
    pub median_nonzero_abs: Option<f64>,
    pub q99: Option<f64>,
    pub robust_clipped_count: Option<u64>,
    pub exact_zero_count: Option<u64>,
    pub unique_bit_patterns: Option<usize>,
    pub calibration_method: Option<&'a str>,
    pub signed_percentile_status: &'static str,
}

pub(super) struct Coverage {
    pub statistics_complete: bool,
    pub values_streamed: usize,
    pub calibrated_tensors: usize,
    pub active_tensor: Option<usize>,
    pub all_requested: bool,
    pub calibration_error: Option<&'static str>,
    pub materialized_tiles: usize,
    pub materialized_bytes: u64,
    pub supported_tensors: usize,
    pub unavailable_tensors: usize,
}

pub(super) struct Status<'a> {
    pub source_identity: &'a str,
    pub model_identity: String,
    pub global_max: Option<f64>,
    pub calibration_complete: bool,
    pub coverage: Coverage,
    // Selected-tensor conversion remains before tile-cache acquisition in the caller.
    pub tensor_status: Option<Value>,
    pub global_calibration_supported: bool,
    pub supported_calibration_complete: bool,
}

fn optional<T: Into<Value>>(value: Option<T>) -> Value {
    value.map_or(Value::Null, Into::into)
}

fn object<const N: usize>(fields: [(&str, Value); N]) -> Value {
    Value::Object(Map::from_iter(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    ))
}

impl From<TensorStatus<'_>> for Value {
    fn from(status: TensorStatus<'_>) -> Self {
        object([
            ("id", Value::from(status.id)),
            (
                "calibration_complete",
                Value::from(status.calibration_complete),
            ),
            ("max_abs", optional(status.max_abs)),
            ("median_nonzero_abs", optional(status.median_nonzero_abs)),
            ("q99", optional(status.q99)),
            (
                "robust_clipped_count",
                optional(status.robust_clipped_count),
            ),
            ("exact_zero_count", optional(status.exact_zero_count)),
            ("unique_bit_patterns", optional(status.unique_bit_patterns)),
            ("calibration_method", optional(status.calibration_method)),
            (
                "rule_status",
                object([(
                    "tensor_signed_percentile",
                    Value::from(status.signed_percentile_status),
                )]),
            ),
        ])
    }
}

impl From<Coverage> for Value {
    fn from(coverage: Coverage) -> Self {
        object([
            (
                "statistics_complete",
                Value::from(coverage.statistics_complete),
            ),
            ("values_streamed", Value::from(coverage.values_streamed)),
            (
                "calibrated_tensors",
                Value::from(coverage.calibrated_tensors),
            ),
            ("active_tensor", optional(coverage.active_tensor)),
            ("all_requested", Value::from(coverage.all_requested)),
            ("calibration_error", optional(coverage.calibration_error)),
            (
                "materialized_tiles",
                Value::from(coverage.materialized_tiles),
            ),
            (
                "materialized_bytes",
                Value::from(coverage.materialized_bytes),
            ),
            ("supported_tensors", Value::from(coverage.supported_tensors)),
            (
                "unavailable_tensors",
                Value::from(coverage.unavailable_tensors),
            ),
        ])
    }
}

impl From<Status<'_>> for Value {
    fn from(status: Status<'_>) -> Self {
        object([
            ("api_version", Value::from(1)),
            ("model_status_version", Value::from(1)),
            ("source_identity", Value::from(status.source_identity)),
            ("model_identity", Value::from(status.model_identity)),
            ("global_max", optional(status.global_max)),
            (
                "calibration_complete",
                Value::from(status.calibration_complete),
            ),
            ("coverage", Value::from(status.coverage)),
            ("tensor_status", optional(status.tensor_status)),
            (
                "global_calibration_supported",
                Value::from(status.global_calibration_supported),
            ),
            (
                "supported_calibration_complete",
                Value::from(status.supported_calibration_complete),
            ),
        ])
    }
}

// Model records are separate from the bounded status projection.
pub(super) struct ModelTensor<'a> {
    pub tensor: Value,
    pub calibration_complete: bool,
    pub slice_required: bool,
    pub display_axes: Vec<usize>,
    pub max_abs: Option<f64>,
    pub median_nonzero_abs: Option<f64>,
    pub q99: Option<f64>,
    pub quantile_order_statistics: Option<&'static str>,
    pub quantile_interpolation: Option<&'static str>,
    pub robust_clipped_count: Option<u64>,
    pub signed_percentile_status: &'static str,
    pub exact_zero_count: Option<u64>,
    pub unique_bit_patterns: Option<usize>,
    pub calibration_method: Option<&'a str>,
}

pub(super) struct ModelCoverage<'a> {
    pub sha_hashed_shards: usize,
    pub sha_expected_matched_shards: usize,
    pub sha_missing_expected_shards: usize,
    pub sha_verified_shards: usize,
    pub statistics_complete: bool,
    pub values_streamed: usize,
    pub calibrated_tensors: usize,
    pub active_tensor: Option<usize>,
    pub calibration_error: &'a Option<String>,
    pub all_requested: bool,
    pub materialized_tiles: usize,
    pub materialized_bytes: u64,
    pub cache_budget_bytes: u64,
    pub fine_tile_file_cap: usize,
    pub supported_tensors: usize,
    pub unavailable_tensors: usize,
}

pub(super) struct Model<'a> {
    pub name: &'a str,
    pub revision: &'a str,
    pub parameter_count: usize,
    pub global_max: Option<f64>,
    pub calibration_complete: bool,
    pub source_directory: &'a std::path::Path,
    pub source_bytes: u64,
    pub header_bytes_read: u64,
    pub source_identity: &'a str,
    pub model_identity: String,
    pub identity_validation: &'a str,
    pub fresh_source_hashes: &'a Option<Value>,
    pub catalog: &'a [Value],
    pub rules: Vec<Value>,
    pub coverage: ModelCoverage<'a>,
    pub global_calibration_supported: bool,
    pub global_calibration_unavailable_reason: Option<&'static str>,
    pub supported_calibration_complete: bool,
}

impl From<ModelTensor<'_>> for Value {
    fn from(tensor: ModelTensor<'_>) -> Self {
        let mut v = tensor.tensor;
        v["calibration_complete"] = serde_json::json!(tensor.calibration_complete);
        v["slice_required"] = serde_json::json!(tensor.slice_required);
        v["display_axes"] = serde_json::json!(tensor.display_axes);
        v["max_abs"] = serde_json::json!(tensor.max_abs);
        v["median_nonzero_abs"] = serde_json::json!(tensor.median_nonzero_abs);
        v["q99"] = serde_json::json!(tensor.q99);
        v["quantile_order_statistics"] = serde_json::json!(tensor.quantile_order_statistics);
        v["quantile_interpolation"] = serde_json::json!(tensor.quantile_interpolation);
        v["robust_clipped_count"] = serde_json::json!(tensor.robust_clipped_count);
        v["rule_status"] =
            serde_json::json!({"tensor_signed_percentile":tensor.signed_percentile_status});
        v["exact_zero_count"] = serde_json::json!(tensor.exact_zero_count);
        v["unique_bit_patterns"] = serde_json::json!(tensor.unique_bit_patterns);
        v["calibration_method"] = serde_json::json!(tensor.calibration_method);
        v
    }
}

impl From<ModelCoverage<'_>> for Value {
    fn from(coverage: ModelCoverage<'_>) -> Self {
        serde_json::json!({
            "source_complete":true,
            "sha_hashed_shards":coverage.sha_hashed_shards,
            "sha_expected_matched_shards":coverage.sha_expected_matched_shards,
            "sha_missing_expected_shards":coverage.sha_missing_expected_shards,
            "sha_verified_shards":coverage.sha_verified_shards,
            "source_validation":"Complete header/index coverage; file identity checked. Freshly hashed shards and matches against saved local expectations are counted separately; saved expectations are not newly authenticated upstream.",
            "statistics_complete":coverage.statistics_complete,
            "values_streamed":coverage.values_streamed,
            "calibrated_tensors":coverage.calibrated_tensors,
            "active_tensor":coverage.active_tensor,
            "calibration_error":coverage.calibration_error,
            "all_requested":coverage.all_requested,
            "materialized_tiles":coverage.materialized_tiles,
            "materialized_bytes":coverage.materialized_bytes,
            "cache_budget_bytes":coverage.cache_budget_bytes,
            "fine_tile_file_cap":coverage.fine_tile_file_cap,
            "all_pixels_materialized":false,
            "rendering_policy":"On-demand numeric tiles. No full-model pixel pyramid required.",
            "supported_tensors":coverage.supported_tensors,
            "unavailable_tensors":coverage.unavailable_tensors,
        })
    }
}

impl From<Model<'_>> for Value {
    fn from(model: Model<'_>) -> Self {
        let mut result = serde_json::json!({
            "api_version":1,
            "extensions":["progressive-calibration-v1","source-dtypes-v1","trailing-slices-v1","source-binding-v2"],
            "backend":"Rust",
            "name":model.name,
            "revision":model.revision,
            "representation":"Original BF16/F16/F32 source values; no model execution",
            "parameter_count":model.parameter_count,
            "global_max":model.global_max,
            "calibration_complete":model.calibration_complete,
            "source_directory":model.source_directory,
            "source_bytes":model.source_bytes,
            "header_bytes_read":model.header_bytes_read,
            "source_identity":model.source_identity,
            "model_identity":model.model_identity,
            "identity_validation":model.identity_validation,
            "fresh_source_hashes":model.fresh_source_hashes,
            "catalog":model.catalog,
            "rules":model.rules,
            "coverage":Value::from(model.coverage),
            "render_semantics":"Each pooled pixel is the F64 mean of the pointwise transformed field over aligned source blocks. Only real edge addresses count. Summation/libm rounding can differ from Python; raw source values are unchanged.",
        });
        result["global_calibration_supported"] =
            serde_json::json!(model.global_calibration_supported);
        result["global_calibration_unavailable_reason"] =
            serde_json::json!(model.global_calibration_unavailable_reason);
        result["supported_calibration_complete"] =
            serde_json::json!(model.supported_calibration_complete);
        result
    }
}
