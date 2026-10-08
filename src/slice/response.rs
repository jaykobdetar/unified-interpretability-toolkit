//! Typed slice responses at the existing `Value` boundary. Identity inputs stay separate.
use serde_json::{Map, Value};

pub(super) struct Selection<'a> {
    pub leading_indices: &'a [usize],
    pub display_axes: Vec<usize>,
}

pub(super) struct Binding<'a> {
    pub model_identity: &'a str,
    pub source_identity: &'a str,
    pub tensor: usize,
    pub name: &'a str,
    pub dtype: &'a str,
    pub shape: &'a [usize],
    pub rows: usize,
    pub cols: usize,
    pub slice: Selection<'a>,
}

pub(super) struct Descriptor<'a> {
    pub identity: &'a str,
    pub leading_indices: &'a [usize],
    pub native_shape: &'a [usize],
    pub display_axes: Vec<usize>,
    pub element_start: usize,
    pub count: usize,
    pub byte_offset: u64,
}

fn array(values: &[usize]) -> Value {
    Value::Array(values.iter().copied().map(Value::from).collect())
}

fn object<const N: usize>(fields: [(&str, Value); N]) -> Value {
    Value::Object(Map::from_iter(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    ))
}

impl From<Selection<'_>> for Value {
    fn from(selection: Selection<'_>) -> Self {
        object([
            ("leading_indices", array(selection.leading_indices)),
            ("display_axes", array(&selection.display_axes)),
        ])
    }
}

impl From<Binding<'_>> for Value {
    fn from(binding: Binding<'_>) -> Self {
        object([
            ("version", Value::from(2)),
            ("model_identity", Value::from(binding.model_identity)),
            ("source_identity", Value::from(binding.source_identity)),
            ("tensor", Value::from(binding.tensor)),
            ("name", Value::from(binding.name)),
            ("dtype", Value::from(binding.dtype)),
            ("shape", array(binding.shape)),
            ("rows", Value::from(binding.rows)),
            ("cols", Value::from(binding.cols)),
            ("slice", Value::from(binding.slice)),
        ])
    }
}

impl From<Descriptor<'_>> for Value {
    fn from(descriptor: Descriptor<'_>) -> Self {
        object([
            ("version", Value::from(1)),
            ("identity", Value::from(descriptor.identity)),
            ("leading_indices", array(descriptor.leading_indices)),
            ("native_shape", array(descriptor.native_shape)),
            ("display_axes", array(&descriptor.display_axes)),
            ("element_start", Value::from(descriptor.element_start)),
            ("count", Value::from(descriptor.count)),
            ("byte_offset", Value::from(descriptor.byte_offset)),
        ])
    }
}

pub(super) fn display_axes(rank: usize) -> Vec<usize> {
    if rank == 1 {
        vec![0]
    } else {
        vec![rank - 2, rank - 1]
    }
}
