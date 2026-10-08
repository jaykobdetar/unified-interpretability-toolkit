//! Inert rank, histogram, palette and mapping-cache compatibility witnesses.
#[path = "support/workspace.rs"]
mod workspace;

use serde_json::json;
use std::{path::PathBuf, sync::Arc};
use weight_atlas_rust::{render, source::Dtype, state::State};

struct Fixture {
    state: State,
    root: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "atlas-rule-boundary-{label}-{}",
            std::process::id()
        ));
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let header = json!({
            "a": {"dtype":"BF16","shape":[4],"data_offsets":[0,8]},
            "b": {"dtype":"BF16","shape":[4],"data_offsets":[8,16]}
        })
        .to_string();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        for word in [0u16, 0x3f80, 0x3f80, 0x4000, 0, 0, 0x3f80, 0x4000] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        std::fs::write(model.join("model.safetensors"), file).unwrap();
        let state = State::open(&model, &root.join("cache"), None, None).unwrap();
        Self { state, root }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn exact_rank_vectors_preserve_ties_sign_and_both_zeros() {
    for (dtype, one, two, four, eight) in [
        (Dtype::Bf16, 0x3f80, 0x4000, 0x4080, 0x4100),
        (Dtype::F16, 0x3c00, 0x4000, 0x4400, 0x4800),
    ] {
        let mut hist = vec![0; 65536];
        hist[0] = 1;
        hist[32768] = 1;
        hist[one] = 2;
        hist[one + 32768] = 1;
        for word in [two, four, eight] {
            hist[word] = 1;
        }
        let result = render::percentile_mapping(dtype, &hist, 8);
        assert!(result.is_ok(), "valid exact histogram");
        let mapping = result.unwrap();
        for (word, expected) in [
            (one, 7. / 16.),
            (two, 11. / 16.),
            (four, 13. / 16.),
            (eight, 15. / 16.),
        ] {
            assert_eq!(mapping.value(word as u32), expected, "positive mid-CDF");
            assert_eq!(
                mapping.value((word + 32768) as u32),
                -expected,
                "negative mid-CDF"
            );
        }
        for word in [0, 32768] {
            assert_eq!(mapping.value(word).to_bits(), 0f64.to_bits(), "rank zero");
        }
    }
}

#[test]
fn sequential_palette_vectors_preserve_clamping_and_ties_even() {
    let field = [-2., -0., 0., 0.5, 1., 2.];
    let expected = [
        247, 244, 249, 255, 247, 244, 249, 255, 247, 244, 249, 255, 166, 142, 196, 255, 84, 39,
        143, 255, 84, 39, 143, 255,
    ];
    for rule in ["tensor_magnitude", "tensor_magnitude_asinh"] {
        assert_eq!(
            render::rgba_for_rule(rule, &field),
            expected,
            "sequential palette"
        );
    }
    for rule in render::RULES {
        if !["tensor_magnitude", "tensor_magnitude_asinh"].contains(&rule) {
            assert_eq!(render::rgba_for_rule(rule, &field), render::rgba(&field));
        }
    }
    // The public palette helper historically falls back for unrecognized IDs.
    assert_eq!(
        render::rgba_for_rule("unknown", &field),
        render::rgba(&field)
    );
}

#[test]
fn tensor_mapping_preserves_uncalibrated_linear_and_rank_admission() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("admission");
    let tensor = fixture.state.source.tensor(0).unwrap();
    let result = fixture
        .state
        .tensor_mapping(tensor, "tensor_linear", &json!({"max":2.}));
    assert!(result.is_ok(), "linear mapping does not load histogram");
    assert_eq!(result.unwrap().value(0x3f80), 0.5);
    let result =
        fixture
            .state
            .tensor_mapping(tensor, "tensor_signed_percentile", &json!({"max":1.}));
    assert_eq!(
        result.err().map(|e| e.to_string()),
        Some("Complete selected-tensor calibration is not ready".into())
    );
}

#[test]
fn tensor_mapping_preserves_exact_legend_histogram_refusal() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("digest");
    fixture.state.calibrate_one(0).unwrap();
    let tensor = fixture.state.source.tensor(0).unwrap();
    let mut legend = render::legend(
        "tensor_signed_percentile",
        fixture.state.stats(0).as_ref(),
        None,
    )
    .unwrap();
    legend["histogram_sha256"] = json!("different-histogram");
    let result = fixture
        .state
        .tensor_mapping(tensor, "tensor_signed_percentile", &legend);
    assert_eq!(
        result.err().map(|e| e.to_string()),
        Some("Percentile calibration mismatch".into()),
        "legend histogram binding"
    );
}

#[test]
fn tensor_mapping_cache_preserves_each_original_distribution() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("distribution");
    let mut mappings = Vec::new();
    for id in [0, 1] {
        fixture.state.calibrate_one(id).unwrap();
        let tensor = fixture.state.source.tensor(id).unwrap();
        let legend = render::legend(
            "tensor_signed_percentile",
            fixture.state.stats(id).as_ref(),
            None,
        )
        .unwrap();
        let first = fixture
            .state
            .tensor_mapping(tensor, "tensor_signed_percentile", &legend)
            .unwrap();
        let second = fixture
            .state
            .tensor_mapping(tensor, "tensor_signed_percentile", &legend)
            .unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "mapping cache reuses exact binding"
        );
        mappings.push(first);
    }
    assert_eq!(mappings[0].value(0x3f80), 0.5, "first distribution");
    assert_eq!(mappings[1].value(0x3f80), 0.625, "second distribution");
    assert!(!Arc::ptr_eq(&mappings[0], &mappings[1]));
}

#[test]
fn tensor_mapping_refuses_missing_histogram_before_binding_or_cache() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("missing-histogram");
    fixture.state.calibrate_one(0).unwrap();
    fixture
        .state
        .calibration
        .lock()
        .unwrap()
        .tensors
        .get_mut(&0)
        .unwrap()
        .histogram_sha256 = None;
    let tensor = fixture.state.source.tensor(0).unwrap();
    let result = fixture.state.tensor_mapping(
        tensor,
        "tensor_signed_percentile",
        &json!({"max":1.,"histogram_sha256":"different-histogram"}),
    );
    let error = result.err().expect("missing histogram must refuse");
    assert!(
        matches!(&error, weight_atlas_rust::Error::Refusal(_)),
        "missing histogram native refusal family"
    );
    assert_eq!(
        error.to_string(),
        "Exact percentile histogram is not ready",
        "missing histogram readiness before legend binding"
    );
    assert_eq!(
        format!("{error:?}"),
        "\"Exact percentile histogram is not ready\"",
        "missing histogram debug spelling"
    );
    assert!(error.source().is_none(), "missing histogram source chain");
    assert!(
        fixture.state.lookups.lock().unwrap().is_empty(),
        "missing histogram does not populate mapping cache"
    );
}
