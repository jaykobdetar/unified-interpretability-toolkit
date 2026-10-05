#[path = "support/workspace.rs"]
mod workspace;

use std::path::PathBuf;
use weight_atlas_rust::{render, server, source::Dtype, state::State};

fn fixture(label: &str, dtype: Dtype, tensors: &[(&str, Vec<u32>)]) -> (PathBuf, State) {
    let root = std::env::temp_dir().join(format!("atlas-rules-{label}-{}", std::process::id()));
    let model = root.join("model");
    std::fs::create_dir_all(&model).unwrap();
    let mut header = serde_json::Map::new();
    let mut raw = Vec::new();
    for (name, words) in tensors {
        let start = raw.len();
        for word in words {
            raw.extend_from_slice(&word.to_le_bytes()[..dtype.bytes()]);
        }
        header.insert((*name).into(),serde_json::json!({"dtype":dtype.name(),"shape":[words.len()],"data_offsets":[start,raw.len()]}));
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend(raw);
    std::fs::write(model.join("model.safetensors"), file).unwrap();
    let state = State::open(&model, &root.join("cache"), None, None).unwrap();
    (root, state)
}
#[test]
fn signed_rank_is_absolute_mid_cdf_with_ties_and_both_zeros() {
    let _isolation = workspace::guard();
    for (label, dtype, words) in [
        (
            "bf",
            Dtype::Bf16,
            vec![0xc080, 0xbf80, 0x8000, 0, 0x3f80, 0x3f80, 0x4000, 0x4100],
        ),
        (
            "half",
            Dtype::F16,
            vec![0xc400, 0xbc00, 0x8000, 0, 0x3c00, 0x3c00, 0x4000, 0x4800],
        ),
    ] {
        let (root, state) = fixture(label, dtype, &[("a", words.clone())]);
        state.calibrate_one(0).unwrap();
        let t = state.source.tensor(0).unwrap();
        let stats = state.stats(0).unwrap();
        let legend = render::legend("tensor_signed_percentile", Some(&stats), None).unwrap();
        assert_eq!(legend["min"], -1.);
        assert_eq!(legend["max"], 1.);
        assert!(legend["units"].as_str().unwrap().contains("dimensionless"));
        let lut = state
            .tensor_mapping(t, "tensor_signed_percentile", &legend)
            .unwrap();
        let values = words.iter().map(|&b| dtype.value(b)).collect::<Vec<_>>();
        let expected = values
            .iter()
            .map(|&v| {
                if v == 0. {
                    0.
                } else {
                    let lo = values.iter().filter(|x| x.abs() < v.abs()).count();
                    let hi = values.iter().filter(|x| x.abs() <= v.abs()).count();
                    v.signum() * (lo + hi) as f64 / (2 * values.len()) as f64
                }
            })
            .collect::<Vec<_>>();
        for (&bits, &want) in words.iter().zip(&expected) {
            assert_eq!(lut.value(bits), want);
        }
        assert_eq!(lut.value(words[1]), -lut.value(words[4]));
        assert_eq!(lut.value(0x8000).to_bits(), 0f64.to_bits());
        let (pooled, metrics) =
            render::tile_fields(&state.source, t, &[&lut], t.max_level - 1, 0, 0).unwrap();
        assert_eq!(metrics.factor, 2);
        for (got, pair) in pooled[0].iter().zip(expected.chunks(2)) {
            assert_eq!(*got, (pair[0] + pair[1]) / 2.);
        }
        let robust = render::legend("tensor_robust99", Some(&stats), None).unwrap();
        assert!((stats.q99 - 7.72).abs() < 1e-14);
        assert_eq!(stats.robust_clipped_count, 1);
        assert_eq!(robust["clipped_fraction"], 0.125);
        let raw = server::inspect(
            &state,
            &server::query(&[
                ("col".into(), "1".into()),
                ("left".into(), "tensor_signed_percentile".into()),
                ("right".into(), "tensor_robust99".into()),
            ]),
        )
        .unwrap();
        assert_eq!(raw["raw_exact"], "-1");
        assert_eq!(raw["transformed"]["left"], expected[1]);
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn robust_zero_quantile_keeps_divisor_one_and_exact_strict_clipping_count() {
    let _isolation = workspace::guard();
    for (label, value) in [("small", 0.5f32), ("large", 2f32), ("zero", 0f32)] {
        let mut words = vec![0; 1001];
        words[1000] = value.to_bits();
        let (root, state) = fixture(label, Dtype::F32, &[("a", words)]);
        state.calibrate_one(0).unwrap();
        let stats = state.stats(0).unwrap();
        assert_eq!(stats.q99, 0.);
        assert_eq!(stats.robust_clipped_count, u64::from(value > 1.));
        let legend = render::legend("tensor_robust99", Some(&stats), None).unwrap();
        assert_eq!(legend["q99"], 0.);
        assert_eq!(legend["effective_divisor"], 1.);
        assert_eq!(legend["min"], -1.);
        assert_eq!(legend["max"], 1.);
        assert_eq!(legend["zero_quantile_fallback"], true);
        let lut = state
            .tensor_mapping(state.source.tensor(0).unwrap(), "tensor_robust99", &legend)
            .unwrap();
        assert_eq!(lut.value(value.to_bits()), (value as f64).min(1.));
        assert!(state
            .tile(0, "tensor_signed_percentile", 0, 0, 0)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported for F32"));
        let raw = server::inspect(
            &state,
            &server::query(&[("left".into(), "tensor_signed_percentile".into())]),
        )
        .unwrap();
        assert_eq!(raw["raw_exact"], "0.0");
        assert!(raw["transformed"]["left"].is_null());
        assert!(raw["transform_errors"]["left"]
            .as_str()
            .unwrap()
            .contains("exact absolute-value rank index"));
        assert!(
            state.model().unwrap()["catalog"][0]["rule_status"]["tensor_signed_percentile"]
                .as_str()
                .unwrap()
                .contains("unsupported")
        );
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn f32_robust_quantile_gaps_and_ties_have_exact_clipped_counts() {
    let _isolation = workspace::guard();
    for (label, values) in [
        ("gap", vec![0f32, 1., 2., 2., 10.]),
        ("ties", vec![1f32; 100]),
        ("subnormal", vec![0., f32::from_bits(1)]),
    ] {
        let words = values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let (root, state) = fixture(label, Dtype::F32, &[("a", words)]);
        state.calibrate_one(0).unwrap();
        let s = state.stats(0).unwrap();
        let divisor = if s.q99 == 0. { 1. } else { s.q99 };
        assert_eq!(
            s.robust_clipped_count,
            values
                .iter()
                .filter(|&&v| (v as f64).abs() > divisor)
                .count() as u64
        );
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn percentile_lut_and_tile_cache_keys_bind_the_distribution() {
    let _isolation = workspace::guard();
    let (root, state) = fixture(
        "distribution",
        Dtype::Bf16,
        &[
            ("a", vec![0, 0x3f80, 0x3f80, 0x4000]),
            ("b", vec![0, 0, 0x3f80, 0x4000]),
        ],
    );
    for id in [0, 1] {
        state.calibrate_one(id).unwrap();
    }
    let mut maps = Vec::new();
    for id in [0, 1] {
        let t = state.source.tensor(id).unwrap();
        let l = render::legend("tensor_signed_percentile", state.stats(id).as_ref(), None).unwrap();
        maps.push(
            state
                .tensor_mapping(t, "tensor_signed_percentile", &l)
                .unwrap(),
        );
    }
    assert_ne!(maps[0].value(0x3f80), maps[1].value(0x3f80));
    let first = state.tile(0, "tensor_signed_percentile", 2, 0, 0).unwrap();
    assert!(!first.1);
    assert!(
        state
            .tile(0, "tensor_signed_percentile", 2, 0, 0)
            .unwrap()
            .1
    );
    assert_ne!(
        first.0,
        state
            .tile(1, "tensor_signed_percentile", 2, 0, 0)
            .unwrap()
            .0
    );
    drop(state);
    let state = State::open(&root.join("model"), &root.join("cache"), None, None).unwrap();
    assert!(
        state
            .tile(0, "tensor_signed_percentile", 2, 0, 0)
            .unwrap()
            .1
    );
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn corrupt_histogram_refuses_rank_but_raw_inspection_remains_available() {
    let _isolation = workspace::guard();
    let (root, state) = fixture("corrupt", Dtype::Bf16, &[("a", vec![0, 0x3f80])]);
    state.calibrate_one(0).unwrap();
    let identity = state.source.identity.clone();
    drop(state);
    std::fs::write(
        root.join("cache/histograms")
            .join(format!("{identity}-0000.u64le.zlib")),
        b"corrupt",
    )
    .unwrap();
    let state = State::open(&root.join("model"), &root.join("cache"), None, None).unwrap();
    assert!(state.tile(0, "tensor_signed_percentile", 1, 0, 0).is_err());
    let raw = server::inspect(
        &state,
        &server::query(&[
            ("col".into(), "1".into()),
            ("left".into(), "tensor_signed_percentile".into()),
            ("right".into(), "tensor_linear".into()),
        ]),
    )
    .unwrap();
    assert_eq!(raw["raw_exact"], "1");
    assert!(raw["transformed"]["left"].is_null());
    assert_eq!(raw["transformed"]["right"], 1.);
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
}
