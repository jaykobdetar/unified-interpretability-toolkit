//! Private CLI presentation records; sample diagnostics inside original JSON evaluation.
use serde_json::{json, Value};
use std::time::Instant;
use weight_atlas_rust::{peak_rss_mib, render::Metrics, source::Source};

pub(super) struct Calibration<'a> {
    pub model: &'a Value,
    pub start: Instant,
    pub minimum: u64,
    pub cpu: usize,
}

impl From<Calibration<'_>> for Value {
    fn from(report: Calibration<'_>) -> Self {
        let Calibration {
            model,
            start,
            minimum,
            cpu,
        } = report;
        json!({"model":model,"wall_seconds":start.elapsed().as_secs_f64(),"peak_rss_mib":peak_rss_mib(),"minimum_available_gib":minimum as f64/1024f64.powi(3),"cpu":cpu})
    }
}

pub(super) struct Tile<'a> {
    pub m: &'a Metrics,
    pub start: Instant,
}

impl From<Tile<'_>> for Value {
    fn from(report: Tile<'_>) -> Self {
        let Tile { m, start } = report;
        json!({"metrics":m,"seconds":start.elapsed().as_secs_f64(),"peak_rss_mib":peak_rss_mib()})
    }
}

pub(super) struct Benchmark<'a> {
    pub records: &'a [Value],
    pub start: Instant,
    pub cpu: usize,
}

impl From<Benchmark<'_>> for Value {
    fn from(report: Benchmark<'_>) -> Self {
        let Benchmark {
            records,
            start,
            cpu,
        } = report;
        json!({"records":records,"wall_seconds":start.elapsed().as_secs_f64(),"peak_rss_mib":peak_rss_mib(),"cpu":cpu})
    }
}

pub(super) struct Metadata<'a> {
    pub s: &'a Source,
    pub start: Instant,
    pub cpu: usize,
}

impl From<Metadata<'_>> for Value {
    fn from(report: Metadata<'_>) -> Self {
        let Metadata { s, start, cpu } = report;
        json!({"source_directory":s.root,"source_identity":s.identity,"header_bytes":s.header_bytes,"source_bytes":s.bytes,"tensor_count":s.tensors.len(),"parameter_count":s.tensors.iter().map(|t|t.count).sum::<usize>(),"catalog":s.tensors,"shards":s.shards,"elapsed_seconds":start.elapsed().as_secs_f64(),"peak_rss_mib":peak_rss_mib(),"cpu":cpu,"full_sha_recomputed":false})
    }
}

pub(super) struct Verification<'a> {
    pub source_identity: &'a str,
    pub records: &'a [Value],
    pub start: Instant,
    pub minimum: u64,
    pub cpu: usize,
}

impl From<Verification<'_>> for Value {
    fn from(report: Verification<'_>) -> Self {
        let Verification {
            source_identity,
            records,
            start,
            minimum,
            cpu,
        } = report;
        json!({"source_identity":source_identity,"shards":records,"wall_seconds":start.elapsed().as_secs_f64(),"peak_rss_mib":peak_rss_mib(),"minimum_available_gib":minimum as f64/1024f64.powi(3),"cpu":cpu,"scope":"Fresh full source SHA; comparisons use explicitly selected local metadata, not a new remote trust check"})
    }
}
