//! Private comparison startup diagnostics; no transport ownership.
use serde_json::Value;

pub(super) struct Startup<'a> {
    pub state: &'a crate::comparison::Comparison,
    pub port: u16,
}

impl From<Startup<'_>> for Value {
    fn from(report: Startup<'_>) -> Self {
        let Startup { state, port } = report;
        serde_json::json!({"listening":format!("http://127.0.0.1:{port}"),"comparison_identity":state.identity,"coordinate_space":crate::comparison::VERSION,"inference_editable":false,"tensors":state.pairs.len(),"peak_rss_mib":crate::peak_rss_mib()})
    }
}

pub(super) enum CalibrationReply {
    Complete { identity: Value },
    Queued { identity: Value, id: usize },
}

impl From<CalibrationReply> for Value {
    fn from(report: CalibrationReply) -> Self {
        match report {
            CalibrationReply::Complete { identity: mut v } => {
                v["complete"] = serde_json::json!(true);
                v
            }
            CalibrationReply::Queued {
                identity: mut v,
                id,
            } => {
                v["queued"] = serde_json::json!(id);
                v
            }
        }
    }
}
