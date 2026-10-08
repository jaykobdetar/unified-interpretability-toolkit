//! Private hosted presentation; retain original expression and channel framing order.
use crate::{api, server::Query, slice::TensorSlice, source::Tensor, state::State};
use serde_json::{json, Value};

pub(super) struct Binding<'a> {
    pub state: &'a State,
    pub slice: &'a TensorSlice,
}
impl From<Binding<'_>> for Value {
    fn from(report: Binding<'_>) -> Self {
        let Binding { state, slice } = report;
        json!({"api_version":1,"source_binding":state.slice_binding(slice)})
    }
}

pub(super) struct Selection<'a> {
    pub selected: Value,
    pub slice: &'a TensorSlice,
}
impl From<Selection<'_>> for Value {
    fn from(report: Selection<'_>) -> Self {
        let Selection {
            mut selected,
            slice,
        } = report;
        selected["slice"] = json!(slice.leading);
        selected["slice_identity"] = json!(slice.identity);
        selected["slice_count"] = json!(slice.tensor.count);
        selected
    }
}

pub(super) struct View<'a> {
    pub state: &'a State,
    pub slice: &'a TensorSlice,
    pub t: &'a Tensor,
    pub q: &'a Query,
    pub selected: &'a Value,
}
impl TryFrom<View<'_>> for Value {
    type Error = crate::Error;
    fn try_from(report: View<'_>) -> crate::Result<Self> {
        let View {
            state,
            slice,
            t,
            q,
            selected,
        } = report;
        Ok(
            json!({"api_version":1,"tensor":selected,"source_binding":state.slice_binding(slice),
                    "legends":state.legends(t,q.get(api::parameter::LEFT,"global_linear"),q.get(api::parameter::RIGHT,"global_asinh"))?,
                    "tile_size":256,"overlap":0,"source_values_unchanged":true}),
        )
    }
}

pub(super) struct Calibrated;
impl From<Calibrated> for Value {
    fn from(_: Calibrated) -> Self {
        json!({"api_version":1,"complete":true})
    }
}

pub(super) struct Frame<'a> {
    pub operation_id: &'a str,
    pub status: i32,
    pub mime: &'a str,
    pub body: &'a [u8],
}
impl From<Frame<'_>> for Value {
    fn from(report: Frame<'_>) -> Self {
        let Frame {
            operation_id,
            status,
            mime,
            body,
        } = report;
        json!({"ack":{"version":1,"operation_id":operation_id,
            "complete":true,"numeric_idle":true},"status":status,"mime":mime,"body_bytes":body.len()})
    }
}
