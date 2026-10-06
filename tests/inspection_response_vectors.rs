//! Exact inspection wire fields with tiny original bytes and held response text.
#[path = "support/workspace.rs"]
mod workspace;

use std::path::PathBuf;
use weight_atlas_rust::{server, source::Dtype, state::State};

struct Fixture {
    root: PathBuf,
    state: State,
    payload_offset: usize,
}

impl Fixture {
    fn new(label: &str, dtype: Dtype, shape: &[usize], words: &[u32]) -> Self {
        let root = std::env::temp_dir().join(format!(
            "atlas-inspection-vectors-{label}-{}",
            std::process::id()
        ));
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let header = serde_json::json!({"weights.λ":{
            "dtype":dtype.name(),"shape":shape,"data_offsets":[0,words.len()*dtype.bytes()]
        }})
        .to_string();
        let payload_offset = header.len() + 8;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        for word in words {
            file.extend_from_slice(&word.to_le_bytes()[..dtype.bytes()]);
        }
        std::fs::write(model.join("tiny.safetensors"), file).unwrap();
        let state = State::open(&model, &root.join("cache"), None, None).unwrap();
        Self {
            root,
            state,
            payload_offset,
        }
    }

    fn bind_text(&self, text: &str) -> String {
        text.replace("@source", &self.state.source.identity)
            .replace("@model", &self.state.model_identity())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn inspect_unready_float_classes_and_native_slice_bytes() {
    let _isolation = workspace::guard();
    for (dtype, words, hexes) in [
        (
            Dtype::Bf16,
            [0x8000, 0x3f80, 0x7f80, 0x7fc1],
            ["0080", "803f", "807f", "c17f"],
        ),
        (
            Dtype::F16,
            [0x8000, 0x3c00, 0x7c00, 0x7e01],
            ["0080", "003c", "007c", "017e"],
        ),
        (
            Dtype::F32,
            [0x8000_0000, 0x3f80_0000, 0x7f80_0000, 0x7fc0_0001],
            ["00000080", "0000803f", "0000807f", "0100c07f"],
        ),
    ] {
        let fixture = Fixture::new(dtype.name(), dtype, &[2, 1, 4], &words.repeat(2));
        for col in 0..4 {
            let q = server::query(&[
                ("slice".into(), "1".into()),
                ("col".into(), col.to_string()),
                ("left".into(), "tensor_magnitude".into()),
                ("right".into(), "tensor_linear".into()),
            ]);
            let got = server::inspect(&fixture.state, &q).unwrap().to_string();
            let exact = ["-0.0", "1", "Infinity", "NaN"][col];
            let classification = ["finite", "finite", "positive_infinity", "nan"][col];
            let error = if col < 2 {
                "Complete selected-tensor calibration is not ready"
            } else {
                "Nonfinite source value: transform unavailable"
            };
            let bf16 = if dtype == Dtype::Bf16 {
                format!("\"{}\"", hexes[col])
            } else {
                "null".into()
            };
            let expected = r#"{"api_version":1,"bf16_hex_le":@bf16,"byte_offset":@offset,"classification":"@class","col":@col,"dtype":"@dtype","element_bytes":@width,"native_indices":[1,0,@col],"raw_exact":"@exact","raw_hex_le":"@hex","row":0,"shard":"tiny.safetensors","source_binding":{"cols":4,"dtype":"@dtype","model_identity":"@model","name":"weights.λ","rows":1,"shape":[2,1,4],"slice":{"display_axes":[1,2],"leading_indices":[1]},"source_identity":"@source","tensor":0,"version":2},"tensor":0,"transform_errors":{"left":"@error","right":"@error"},"transformed":{"left":null,"right":null},"transforms_ready":false}"#;
            let expected = fixture
                .bind_text(expected)
                .replace("@bf16", &bf16)
                .replace(
                    "@offset",
                    &(fixture.payload_offset + (4 + col) * dtype.bytes()).to_string(),
                )
                .replace("@class", classification)
                .replace("@col", &col.to_string())
                .replace("@dtype", dtype.name())
                .replace("@width", &dtype.bytes().to_string())
                .replace("@exact", exact)
                .replace("@hex", hexes[col])
                .replace("@error", error);
            assert_eq!(
                got, expected,
                "inspection response bytes {dtype:?} col={col}"
            );
        }
    }
}

#[test]
fn inspect_ready_values_and_validation_order_remain_exact() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("ready", Dtype::Bf16, &[2], &[0x3f80, 0x4000]);
    fixture.state.calibrate_one(0).unwrap();
    let pairs = [
        ("col".into(), "1".into()),
        ("left".into(), "tensor_linear".into()),
        ("right".into(), "tensor_magnitude".into()),
    ];
    let q = server::query(&pairs);
    let expected = r#"{"api_version":1,"bf16_hex_le":"0040","byte_offset":@offset,"classification":"finite","col":1,"dtype":"BF16","element_bytes":2,"native_indices":[1],"raw_exact":"2","raw_hex_le":"0040","row":0,"shard":"tiny.safetensors","source_binding":{"cols":2,"dtype":"BF16","model_identity":"@model","name":"weights.λ","rows":1,"shape":[2],"slice":{"display_axes":[0],"leading_indices":[]},"source_identity":"@source","tensor":0,"version":2},"tensor":0,"transform_errors":{},"transformed":{"left":1.0,"right":1.0},"transforms_ready":true}"#;
    assert_eq!(
        server::inspect(&fixture.state, &q).unwrap().to_string(),
        fixture
            .bind_text(expected)
            .replace("@offset", &(fixture.payload_offset + 2).to_string()),
        "inspection response bytes ready"
    );
    for (col, expected) in [("2", "Address outside tensor"), ("1", "Unknown color rule")] {
        let q = server::query(&[
            ("col".into(), col.into()),
            ("left".into(), "unknown-vector-rule".into()),
        ]);
        assert_eq!(
            server::inspect(&fixture.state, &q).unwrap_err().to_string(),
            expected
        );
    }
}
