//! Independent viewer envelopes over two original BF16 values.
#[path = "support/workspace.rs"]
mod workspace;

use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use weight_atlas_rust::{slice::TensorSlice, state::State};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-response-envelope-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        std::fs::create_dir_all(root.join("model")).unwrap();
        let header = br#"{"weights":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        for bits in [0x3f80u16, 0xc000] {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        std::fs::write(root.join("model/tiny.safetensors"), bytes).unwrap();
        Self(root)
    }

    fn state(&self) -> State {
        let state = State::open(&self.0.join("model"), &self.0.join("cache"), None, None).unwrap();
        state.calibrate_one(0).unwrap();
        state
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn linear_legend(id: &str, scope: &str) -> Value {
    // Held original metadata plus independently known extrema of [1, -2].
    let metadata: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/rule_metadata_vectors.json")).unwrap();
    let mut expected = metadata.into_iter().find(|row| row["id"] == id).unwrap();
    expected.as_object_mut().unwrap().extend(
        json!({
            "min":-2.0,"zero":0.0,"max":2.0,"s":null,"scope":scope,
            "units":"raw weight","clipped_fraction":0.0,
            "color_warning":"Finite 8-bit colors merge nearby weights; neutral does not prove exact zero."
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    expected
}

#[test]
fn viewer_legends_keep_both_distinct_rules_and_exact_envelope() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let state = fixture.state();
    let actual = state
        .legends(&state.source.tensors[0], "tensor_linear", "global_linear")
        .unwrap();
    let expected = json!({
        "left":linear_legend("tensor_linear", "complete original tensor"),
        "right":linear_legend("global_linear", "complete checkpoint")
    });
    assert_eq!(actual.to_string(), expected.to_string());
}

#[test]
fn overview_keeps_exact_envelope_and_published_tile_coordinates() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new();
    let state = fixture.state();
    let rules = ["tensor_linear", "global_linear"];
    let slice = TensorSlice::new(&state.source, 0, &[]).unwrap();
    // Identity helpers have separate vectors; this witnesses their forwarding.
    let binding = state.slice_binding(&slice);
    let tile_bindings = rules.map(|rule| state.tile_binding(0, &[], rule).unwrap());
    let actual = state.prepare_overview(0, &[], &rules, 2).unwrap();
    let mut tiles = Vec::new();
    for (rule, tile_binding) in rules.into_iter().zip(tile_bindings) {
        let (png, hit, _) = state.tile(0, rule, 1, 0, 0).unwrap();
        assert!(hit, "overview must publish the requested tile to the cache");
        tiles.push(json!({
            "rule":rule,"level":1,"x":0,"y":0,
            "binding":tile_binding,"png_bytes":png.len()
        }));
    }
    let expected = json!({
        "api_version":1,"source_binding":binding,"max_values":2,
        "metrics":{"source_count":2,"source_bytes_read":4,"max_raw_band_values":2,
                   "factor":1,"width":2,"height":1},
        "tiles":tiles,
        "coverage":"one selected-slice overview per rule; fine tiles remain on demand"
    });
    assert_eq!(actual.to_string(), expected.to_string());
}
