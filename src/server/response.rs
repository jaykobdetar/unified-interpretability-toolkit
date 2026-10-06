//! Typed inspection fields converted at the existing JSON value boundary.
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
