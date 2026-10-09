//! Private candidate receipt; numerical work, sealing and grants stay in the worker.
use crate::strength::{snapshot::Layout, StrengthProfile};
use serde_json::{json, Value};

pub(super) struct Candidate<'a> {
    pub revision: &'a str,
    pub profile: &'a StrengthProfile,
    pub before: usize,
    pub layout: &'a Layout,
}
impl From<Candidate<'_>> for Value {
    fn from(report: Candidate<'_>) -> Self {
        let Candidate {
            revision,
            profile,
            before,
            layout,
        } = report;
        json!({"schema":"weight-atlas.profile-candidate.v1","revision":revision,
        "visited_values":profile.visited_values(),"new_values":profile.visited_values()-before,
        "frame_bytes":layout.frame_bytes,"live_bytes":layout.live_bytes})
    }
}
