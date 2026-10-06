//! Held format admission, metadata spelling and string deserialization witnesses.
#[path = "support/workspace.rs"]
mod workspace;

use std::path::PathBuf;
use weight_atlas_rust::source::{Dtype, Source, Tensor};

struct Fixture {
    root: PathBuf,
    payload_offset: usize,
}

impl Fixture {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("atlas-format-ingress-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let header = r#"{"a-bf":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b-half":{"dtype":"F16","shape":[1],"data_offsets":[2,4]},"c-single":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"d-unknown":{"dtype":"NEW_CODEC","shape":[1],"data_offsets":[8,10]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0; 10]);
        std::fs::write(root.join("tiny.safetensors"), bytes).unwrap();
        Self {
            root,
            payload_offset: header.len() + 8,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn original_parser_admission_and_exact_refusal() {
    for (name, expected) in [
        ("BF16", Dtype::Bf16),
        ("F16", Dtype::F16),
        ("F32", Dtype::F32),
    ] {
        assert_eq!(
            Dtype::parse(name).ok(),
            Some(expected),
            "format parser admission {name}"
        );
    }
    for name in ["", "bf16", "F64", "I8", "NEW_CODEC", "Ｆ16"] {
        let outcome = Dtype::parse(name)
            .map(|value| format!("{value:?}"))
            .unwrap_or_else(|error| error.to_string());
        assert_eq!(
            outcome, "Only BF16, F16 and F32 tensors are supported",
            "format parser exact refusal {name}"
        );
    }
}

#[test]
fn header_format_names_widths_admission_and_serialized_fields() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let result = Source::open(&fixture.root);
    let outcome = result
        .as_ref()
        .map(|_| "accepted".to_string())
        .unwrap_or_else(|error| error.to_string());
    assert_eq!(outcome, "accepted", "format header admission");
    let source = result.unwrap();
    for (id, name, dtype, width, offset, available, reason) in [
        (0, "a-bf", "BF16", 2, 0, true, "null"),
        (1, "b-half", "F16", 2, 2, true, "null"),
        (2, "c-single", "F32", 4, 4, true, "null"),
        (
            3,
            "d-unknown",
            "NEW_CODEC",
            0,
            8,
            false,
            "\"Unknown storage encoding; declared extent checked only, no numeric interpretation\"",
        ),
    ] {
        let expected = r#"{"available":@available,"byte_offset":@offset,"cols":1,"count":1,"dtype":"@dtype","element_bytes":@width,"id":@id,"max_level":0,"min_level":0,"name":"@name","rows":1,"shape":[1],"shard":"tiny.safetensors","shard_id":0,"unavailable_reason":@reason}"#
            .replace("@available", &available.to_string())
            .replace("@offset", &(fixture.payload_offset + offset).to_string())
            .replace("@dtype", dtype).replace("@width", &width.to_string())
            .replace("@id", &id.to_string()).replace("@name", name).replace("@reason", reason);
        let got = serde_json::to_value(&source.tensors[id])
            .unwrap()
            .to_string();
        assert_eq!(got, expected, "format metadata bytes {name}");
    }
}

#[test]
fn metadata_string_round_trip_keeps_unknown_names_and_json_errors() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let source = Source::open(&fixture.root).unwrap();
    for tensor in &source.tensors {
        let raw = serde_json::to_string(tensor).unwrap();
        let restored: Tensor = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            serde_json::to_string(&restored).unwrap(),
            raw,
            "format string round trip"
        );
        assert_eq!(
            format!("{:?}", restored.dtype),
            format!("{:?}", tensor.dtype)
        );
    }
    let mut value = serde_json::to_value(&source.tensors[0]).unwrap();
    value["dtype"] = serde_json::json!(7);
    let error = serde_json::from_value::<Tensor>(value)
        .unwrap_err()
        .to_string();
    assert_eq!(
        error, "invalid type: integer `7`, expected a string",
        "format JSON error spelling"
    );
}
