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
