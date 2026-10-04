use std::path::{Path, PathBuf};
use weight_atlas_rust::{
    render, server,
    source::{exact_decimal_for, Dtype, Source},
    state::State,
};

fn fixture(label: &str, dtype: Dtype, shape: &[usize], words: &[u32]) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("atlas-dtype-{label}-{}", std::process::id()));
    let model = root.join("model");
    std::fs::create_dir_all(&model).unwrap();
    write(&model, dtype, shape, words);
    (root, model)
}
fn write(model: &Path, dtype: Dtype, shape: &[usize], words: &[u32]) {
    let header = serde_json::json!({"tensor":{"dtype":dtype.name(),"shape":shape,"data_offsets":[0, words.len()*dtype.bytes()]}}).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes()[..dtype.bytes()]);
    }
    std::fs::write(model.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn exhaustive_half_words_decode_against_binary32_normalization() {
    for word in 0..65536u32 {
        // Independent bit-normalization construction of an IEEE binary32 word.
        let sign = (word & 0x8000) << 16;
        let exponent = (word >> 10) & 31;
        let fraction = word & 1023;
        let expanded = match (exponent, fraction) {
            (0, 0) => sign,
            (0, _) => {
                let leading = 31 - fraction.leading_zeros();
                sign | ((leading + 103) << 23) | ((fraction ^ (1 << leading)) << (23 - leading))
            }
            (31, _) => sign | 0x7f80_0000 | (fraction << 13),
            _ => sign | ((exponent + 112) << 23) | (fraction << 13),
        };
        let expected = f32::from_bits(expanded) as f64;
        let got = Dtype::F16.value(word);
        assert_eq!(Dtype::F16.finite(word), expected.is_finite());
        if expected.is_nan() {
            assert!(got.is_nan());
        } else {
            assert_eq!(got.to_bits(), expected.to_bits());
        }
        if got.is_finite() {
            assert_eq!(
                exact_decimal_for(Dtype::F16, word)
                    .parse::<f64>()
                    .unwrap()
                    .to_bits(),
                got.to_bits()
            );
        }
    }
}

#[test]
fn exact_decimals_include_signed_zero_subnormal_and_extremes() {
    assert_eq!(exact_decimal_for(Dtype::F16, 0x8000), "-0.0");
    assert_eq!(
        exact_decimal_for(Dtype::F16, 1),
        "0.000000059604644775390625"
    );
    assert_eq!(exact_decimal_for(Dtype::F16, 0x7bff), "65504");
    assert_eq!(exact_decimal_for(Dtype::F32, 0x8000_0000), "-0.0");
    assert_eq!(
        exact_decimal_for(Dtype::F32, 0x3dcc_cccd),
        "0.100000001490116119384765625"
    );
    assert_eq!(
        exact_decimal_for(Dtype::F32, 0x7f7f_ffff),
        "340282346638528859811704183484516925440"
    );
    assert_eq!(exact_decimal_for(Dtype::F32, 1), "0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125");
}

#[test]
fn exact_raw_inspection_calibration_pooling_and_dtype_lookup_separation() {
    for dtype in [Dtype::Bf16, Dtype::F16, Dtype::F32] {
        let words = match dtype {
            Dtype::Bf16 => vec![0, 0x8000, 0x3f80, 0xbf80, 1, 0x7f7f],
            Dtype::F16 => vec![0, 0x8000, 0x3c00, 0xbc00, 1, 0x7bff],
            Dtype::F32 => vec![0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 1, 0x7f7f_ffff],
        };
        let (root, model) = fixture(dtype.name(), dtype, &[2, 3], &words);
        let state = State::open(&model, &root.join("cache"), None, None).unwrap();
        let t = state.source.tensor(0).unwrap();
        assert_eq!(t.element_bytes, dtype.bytes());
        for (index, &bits) in words.iter().enumerate() {
            let q = server::query(&[
                ("row".into(), (index / 3).to_string()),
                ("col".into(), (index % 3).to_string()),
                ("left".into(), "tensor_magnitude".into()),
                ("right".into(), "tensor_linear".into()),
            ]);
            let got = server::inspect(&state, &q).unwrap();
            assert_eq!(
                got["raw_hex_le"],
                bits.to_le_bytes()[..dtype.bytes()]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
            assert_eq!(got["dtype"], dtype.name());
            assert_eq!(
                got["byte_offset"],
                t.byte_offset + (index * dtype.bytes()) as u64
            );
            assert_eq!(got["raw_exact"], exact_decimal_for(dtype, bits));
            assert_eq!(got["transforms_ready"], false);
            assert_eq!(got["bf16_hex_le"].is_null(), dtype != Dtype::Bf16);
        }
        state.calibrate_one(0).unwrap();
        let stats = state.stats(0).unwrap();
        assert_eq!(stats.exact_zero_count, 2);
        assert_eq!(stats.max_abs, dtype.value(words[5]));
        assert_eq!(stats.median_nonzero_abs, 1.);
        assert_eq!(stats.unique_bit_patterns.is_none(), dtype == Dtype::F32);
        let legend = render::legend("tensor_magnitude", Some(&stats), None).unwrap();
        let mapping = state.mapping("tensor_magnitude", &legend, dtype).unwrap();
        let (fields, m) = render::tile_fields(&state.source, t, &[&mapping], 1, 0, 0).unwrap();
        assert_eq!((m.width, m.height, m.factor), (2, 1, 2));
        let values = words
            .iter()
            .map(|&w| dtype.value(w).abs() / stats.max_abs)
            .collect::<Vec<_>>();
        assert_eq!(
            fields[0],
            [
                (values[0] + values[1] + values[3] + values[4]) / 4.,
                (values[2] + values[5]) / 2.
            ]
        );
        let one = serde_json::json!({"max":1.,"s":null});
        let half = state.mapping("tensor_linear", &one, Dtype::F16).unwrap();
        let bf = state.mapping("tensor_linear", &one, Dtype::Bf16).unwrap();
        assert_eq!(half.value(0x3c00), 1.);
        assert_ne!(half.value(0x3c00), bf.value(0x3c00));
        drop(state);
        let reopened = State::open(&model, &root.join("cache"), None, None).unwrap();
        assert_eq!(reopened.stats(0).unwrap().dtype, dtype.name());
        let restored = reopened.stats(0).unwrap();
        assert_eq!(restored.max_abs.to_bits(), stats.max_abs.to_bits());
        assert_eq!(restored.q99.to_bits(), stats.q99.to_bits());
        assert_eq!(
            restored.median_nonzero_abs.to_bits(),
            stats.median_nonzero_abs.to_bits()
        );
        drop(reopened);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn nonfinite_never_publishes_statistics_or_renders_replacement_pixels() {
    for (label, dtype, word) in [
        ("half-inf", Dtype::F16, 0x7c00),
        ("half-nan", Dtype::F16, 0x7e13),
        ("float-inf", Dtype::F32, 0xff80_0000),
        ("float-nan", Dtype::F32, 0x7fc0_1234),
    ] {
        let (root, model) = fixture(label, dtype, &[1], &[word]);
        let state = State::open(&model, &root.join("cache"), None, None).unwrap();
        let raw = server::inspect(&state, &server::query(&[])).unwrap();
        assert_ne!(raw["classification"], "finite");
        assert_eq!(raw["transformed"]["left"], serde_json::Value::Null);
        assert!(state.calibrate_one(0).is_err());
        assert!(state.stats(0).is_none());
        assert!(!root.join("cache/calibration.json").exists());
        let m = render::mapping(
            "tensor_linear",
            &serde_json::json!({"max":1.,"s":null}),
            dtype,
        )
        .unwrap();
        assert!(render::tile_fields(
            &state.source,
            state.source.tensor(0).unwrap(),
            &[&m],
            0,
            0,
            0
        )
        .is_err());
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn f32_rank_selection_crosses_buckets_and_handles_zeros_and_ties() {
    for (label, words) in [
        (
            "radix",
            vec![
                0,
                0x8000_0000,
                1,
                0x0080_0000,
                0x3f00_0000,
                0x3f80_0000,
                0x3f80_0000,
                0x4000_0000,
                0x7f7f_ffff,
            ],
        ),
        ("zero", vec![0, 0x8000_0000]),
        ("single", vec![1]),
    ] {
        let (root, model) = fixture(label, Dtype::F32, &[words.len()], &words);
        let source = Source::open(&model).unwrap();
        let (stats, hist) = render::calibrate(&source, source.tensor(0).unwrap()).unwrap();
        assert!(hist.is_empty());
        let mut abs = words
            .iter()
            .map(|&w| Dtype::F32.value(w).abs())
            .collect::<Vec<_>>();
        abs.sort_by(f64::total_cmp);
        let q = (abs.len() - 1) as f64 * 0.99;
        assert_eq!(
            stats.q99,
            abs[q.floor() as usize]
                + (abs[q.ceil() as usize] - abs[q.floor() as usize]) * (q - q.floor())
        );
        let positive = abs.into_iter().filter(|&v| v > 0.).collect::<Vec<_>>();
        let median = if positive.is_empty() {
            0.
        } else {
            let n = positive.len();
            if n % 2 == 1 {
                positive[n / 2]
            } else {
                positive[n / 2 - 1] + (positive[n / 2] - positive[n / 2 - 1]) * 0.5
            }
        };
        assert_eq!(stats.median_nonzero_abs, median);
        drop(source);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn width_and_offsets_must_match_dtype_with_no_overflow_or_truncation() {
    let (root, model) = fixture("bad-width", Dtype::F32, &[1], &[0]);
    let p = model.join("model.safetensors");
    let original = std::fs::read(&p).unwrap();
    let header_len = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let mut h: serde_json::Value = serde_json::from_slice(&original[8..8 + header_len]).unwrap();
    for spec in [
        serde_json::json!({"dtype":"F64","shape":[1],"data_offsets":[0,4]}),
        serde_json::json!({"dtype":"F32","shape":[1],"data_offsets":[0,2]}),
        serde_json::json!({"dtype":"F16","shape":[1],"data_offsets":[0,18446744073709551615u64]}),
        serde_json::json!({"dtype":"F32","shape":[18446744073709551615u64,2],"data_offsets":[0,4]}),
    ] {
        h["tensor"] = spec;
        let text = h.to_string();
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        bytes.extend_from_slice(&[0; 4]);
        std::fs::write(&p, bytes).unwrap();
        assert!(Source::open(&model).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn json_scale_roundtrips_preserve_source_and_derived_f64_bits() {
    let mut word = 0x7f7fffffu32;
    for _ in 0..10000 {
        let value = f32::from_bits(word) as f64;
        if value.is_finite() {
            for x in [value, value - 1., value / 3., -0.] {
                let encoded = serde_json::to_string(&x).unwrap();
                let restored: f64 = serde_json::from_str(&encoded).unwrap();
                assert_eq!(x.to_bits(), restored.to_bits(), "{encoded}");
            }
        }
        word = word.wrapping_mul(1664525).wrapping_add(1013904223);
    }
}
