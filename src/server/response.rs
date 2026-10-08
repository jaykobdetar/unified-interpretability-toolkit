//! Typed viewer response fields converted at the existing JSON value boundary.
use serde_json::{Map, Value};

pub(super) struct Inspection<'a> {
    pub source_binding: Value,
    pub tensor: usize,
    pub row: usize,
    pub col: usize,
    pub raw_exact: String,
    pub dtype: &'static str,
    pub element_bytes: usize,
    pub raw_hex_le: &'a str,
    pub classification: &'static str,
    pub bf16_hex_le: Option<&'a str>,
    pub shard: &'a str,
    pub byte_offset: u64,
    pub native_indices: Vec<usize>,
    pub transformed: Map<String, Value>,
    pub transform_errors: Map<String, Value>,
    pub transforms_ready: bool,
}

impl From<Inspection<'_>> for Value {
    fn from(response: Inspection<'_>) -> Self {
        let fields = [
            ("api_version", Value::from(1)),
            ("source_binding", response.source_binding),
            ("tensor", Value::from(response.tensor)),
            ("row", Value::from(response.row)),
            ("col", Value::from(response.col)),
            ("raw_exact", Value::from(response.raw_exact)),
            ("dtype", Value::from(response.dtype)),
            ("element_bytes", Value::from(response.element_bytes)),
            ("raw_hex_le", Value::from(response.raw_hex_le)),
            ("classification", Value::from(response.classification)),
            (
                "bf16_hex_le",
                response.bf16_hex_le.map_or(Value::Null, Value::from),
            ),
            ("shard", Value::from(response.shard)),
            ("byte_offset", Value::from(response.byte_offset)),
            (
                "native_indices",
                Value::Array(
                    response
                        .native_indices
                        .into_iter()
                        .map(Value::from)
                        .collect(),
                ),
            ),
            ("transformed", Value::Object(response.transformed)),
            ("transform_errors", Value::Object(response.transform_errors)),
            ("transforms_ready", Value::from(response.transforms_ready)),
        ];
        Value::Object(Map::from_iter(
            fields
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value)),
        ))
    }
}

pub(super) struct View {
    pub tensor: Value,
    pub source_binding: Value,
    pub legends: Value,
    pub left_binding: String,
    pub right_binding: String,
}

impl From<View> for Value {
    fn from(response: View) -> Self {
        let bindings = Map::from_iter([
            ("left".to_owned(), Value::from(response.left_binding)),
            ("right".to_owned(), Value::from(response.right_binding)),
        ]);
        let fields = [
            ("api_version", Value::from(1)),
            ("tensor", response.tensor),
            ("source_binding", response.source_binding),
            ("legends", response.legends),
            ("tile_bindings", Value::Object(bindings)),
            ("tile_size", Value::from(256)),
            ("overlap", Value::from(0)),
            ("source_values_unchanged", Value::from(true)),
        ];
        Value::Object(Map::from_iter(
            fields
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value)),
        ))
    }
}

pub(super) struct ErrorBody {
    pub error: String,
}

impl From<ErrorBody> for Value {
    fn from(response: ErrorBody) -> Self {
        Value::Object(Map::from_iter([
            ("error".to_owned(), Value::from(response.error)),
            ("api_version".to_owned(), Value::from(1)),
        ]))
    }
}

pub(super) struct CodedErrorBody<'a> {
    pub code: &'a str,
    pub error: &'a str,
}

impl From<CodedErrorBody<'_>> for Value {
    fn from(response: CodedErrorBody<'_>) -> Self {
        Value::Object(Map::from_iter([
            ("api_version".to_owned(), Value::from(1)),
            ("code".to_owned(), Value::from(response.code)),
            ("error".to_owned(), Value::from(response.error)),
        ]))
    }
}

pub(super) enum CalibrationReply<'a> {
    Checkpoint(&'a str),
    Tensor(usize),
    Complete,
}

impl From<CalibrationReply<'_>> for Value {
    fn from(response: CalibrationReply<'_>) -> Self {
        let detail = match response {
            CalibrationReply::Checkpoint(scope) => ("queued".to_owned(), Value::from(scope)),
            CalibrationReply::Tensor(id) => ("queued".to_owned(), Value::from(id)),
            CalibrationReply::Complete => ("complete".to_owned(), Value::from(true)),
        };
        Value::Object(Map::from_iter([
            ("api_version".to_owned(), Value::from(1)),
            detail,
        ]))
    }
}

pub(super) struct Selection<'a> {
    pub tensor: Value,
    pub slice: &'a [usize],
    pub slice_identity: &'a str,
    pub slice_count: usize,
}

impl From<Selection<'_>> for Value {
    fn from(response: Selection<'_>) -> Self {
        let mut tensor = response.tensor;
        tensor["slice"] = serde_json::json!(response.slice);
        tensor["slice_identity"] = serde_json::json!(response.slice_identity);
        tensor["slice_count"] = serde_json::json!(response.slice_count);
        tensor
    }
}

pub(super) struct Startup<'a> {
    pub state: &'a crate::state::State,
    pub port: u16,
}

impl TryFrom<Startup<'_>> for Value {
    type Error = crate::Error;
    fn try_from(report: Startup<'_>) -> crate::Result<Self> {
        let Startup { state, port } = report;
        Ok(
            serde_json::json!({"listening":format!("http://127.0.0.1:{port}"),"metadata_ready_seconds":state.started.elapsed().as_secs_f64(),"header_bytes":state.source.header_bytes,"tensors":state.source.tensors.len(),"peak_rss_mib":crate::peak_rss_mib(),"resources":crate::resources::snapshot()?}),
        )
    }
}
