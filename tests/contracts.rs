use weight_atlas_rust::{
    render::{self, Stats},
    source::{json_unique, value},
    state::TileCache,
};
#[test]
fn duplicate_keys_at_all_levels_are_rejected() {
    assert!(json_unique(br#"{"x":{"y":1,"y":2}}"#).is_err());
    assert!(json_unique(br#"{"x":[{"y":1,"y":2}]}"#).is_err());
}
#[test]
fn all_bf16_format_words_decode_independently() {
    for bits in 0..=65535u16 {
        let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
        let exponent = (bits >> 7) & 255;
        let fraction = bits & 127;
        let got = value(bits);
        if exponent == 255 {
            if fraction == 0 {
                assert!(got.is_infinite())
            } else {
                assert!(got.is_nan())
            }
        } else {
            let expected = if exponent == 0 {
                sign * (fraction as f64) * 2f64.powi(-133)
            } else {
                sign * (1. + fraction as f64 / 128.) * 2f64.powi(exponent as i32 - 127)
            };
            assert_eq!(got, expected);
            if expected == 0. {
                assert_eq!(got.is_sign_negative(), sign < 0.);
            }
        }
    }
}
#[test]
fn zero_calibration_and_exact_palette_endpoints() {
    let s = Stats {
        count: 1,
        max_abs: 0.,
        median_nonzero_abs: 0.,
        q99: 0.,
        robust_clipped_count: 0,
        histogram_sha256: Some("fixture".into()),
        exact_zero_count: 1,
        unique_bit_patterns: Some(1),
        dtype: "BF16".into(),
        calibration_method: "exact-16-bit-histogram".into(),
        seconds: 0.,
    };
    for rule in render::RULES {
        let l = render::legend(rule, Some(&s), Some(0.)).unwrap();
        if rule == "tensor_signed_percentile" {
            let mut hist = vec![0; 65536];
            hist[0] = 1;
            let mapping =
                render::percentile_mapping(weight_atlas_rust::source::Dtype::Bf16, &hist, 1)
                    .unwrap();
            assert_eq!(mapping.value(0), 0.);
            assert_eq!(mapping.value(32768), 0.);
            continue;
        }
        let lut = render::lookup(rule, &l);
        assert_eq!(lut[0], 0.);
        assert_eq!(lut[32768], 0.);
        if rule.ends_with("asinh") {
            assert_eq!(l["s"], 1.);
        }
    }
    assert_eq!(
        render::rgba(&[-1., 0., 1.]),
        [37, 99, 242, 255, 242, 242, 242, 255, 242, 38, 38, 255]
    );
}
#[test]
fn no_global_or_local_legend_before_its_calibration() {
    assert!(render::legend("global_linear", None, None).is_err());
    assert!(render::legend("tensor_linear", None, Some(34.)).is_err());
    assert!(render::legend("bogus", None, Some(34.)).is_err());
}
#[test]
fn fine_cache_file_count_is_bounded() {
    let dir = std::env::temp_dir().join(format!("atlas-cache-contract-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut c = TileCache::open(&dir).unwrap();
    for i in 0..1002 {
        c.put(&format!("{i:064x}"), b"fixture").unwrap()
    }
    assert_eq!(c.entries.len(), 1000);
    assert_eq!(c.bytes, 7000);
    assert!(c.get(&format!("{:064x}", 0)).unwrap().is_none());
    assert_eq!(
        c.get(&format!("{:064x}", 1001)).unwrap().unwrap(),
        b"fixture"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn exact_decimals_preserve_long_expansions_and_signed_zero() {
    use weight_atlas_rust::source::exact_decimal;
    assert_eq!(exact_decimal(0x3dcc), "0.099609375");
    assert_eq!(exact_decimal(0x8000), "-0.0");
    assert_eq!(
        exact_decimal(0x7f7f),
        "338953138925153547590470800371487866880"
    );
    assert_eq!(exact_decimal(0x4208), "34");
    for bits in 0..65536u32 {
        let bits = bits as u16;
        if value(bits).is_finite() {
            assert_eq!(exact_decimal(bits).parse::<f64>().unwrap(), value(bits));
        }
    }
}

fn small_model(label: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("atlas-{label}-{}", std::process::id()));
    let model = root.join("model");
    let cache = root.join("cache");
    std::fs::create_dir_all(&model).unwrap();
    let h = br#"{"tensor":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(h);
    bytes.extend_from_slice(&[0x80, 0x3f, 0x80, 0xbf]);
    std::fs::write(model.join("model.safetensors"), bytes).unwrap();
    (root, model, cache)
}
#[test]
fn failed_calibration_publication_is_retryable_without_false_readiness() {
    use weight_atlas_rust::state::State;
    let (root, model, cache) = small_model("calibration-retry");
    let state = State::open(&model, &cache, None, None).unwrap();
    // Ordinary disposable filesystem fixture: inject a destination rename error.
    std::fs::create_dir(cache.join("calibration.json")).unwrap();
    assert!(state.calibrate_one(0).is_err());
    assert!(state.stats(0).is_none());
    assert!(state.global().is_none());
    assert!(state.progress.lock().unwrap().active.is_none());
    assert!(state.progress.lock().unwrap().error.is_some());
    assert!(std::fs::read_dir(&cache).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".partial-")));
    std::fs::remove_dir(cache.join("calibration.json")).unwrap();
    state.calibrate_one(0).unwrap();
    assert_eq!(state.global(), Some(1.));
    drop(state);
    let reopened = State::open(&model, &cache, None, None).unwrap();
    assert_eq!(reopened.global(), Some(1.));
    drop(reopened);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn active_source_mutation_is_refused_and_saved_statistics_invalidate() {
    use weight_atlas_rust::state::State;
    let (root, model, cache) = small_model("identity");
    let state = State::open(&model, &cache, None, None).unwrap();
    state.calibrate_one(0).unwrap();
    assert_eq!(state.global(), Some(1.));
    let p = model.join("model.safetensors");
    let mut bytes = std::fs::read(&p).unwrap();
    let n = bytes.len();
    bytes[n - 1] = 0xc0;
    std::fs::write(&p, bytes).unwrap();
    assert!(state
        .source
        .scalar(state.source.tensor(0).unwrap(), 0, 0)
        .is_err());
    drop(state);
    let changed = State::open(&model, &cache, None, None).unwrap();
    assert_eq!(changed.global(), None);
    drop(changed);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn corrupted_checksum_and_changed_schema_cannot_claim_calibration() {
    use weight_atlas_rust::{sha, state::State};
    let (root, model, cache) = small_model("cache-schema");
    let state = State::open(&model, &cache, None, None).unwrap();
    state.calibrate_one(0).unwrap();
    drop(state);
    let p = cache.join("calibration.json");
    let original = std::fs::read(&p).unwrap();
    let mut wrapper: serde_json::Value = serde_json::from_slice(&original).unwrap();
    wrapper["sha256"] = serde_json::json!("wrong");
    std::fs::write(&p, wrapper.to_string()).unwrap();
    let state = State::open(&model, &cache, None, None).unwrap();
    assert_eq!(state.global(), None);
    assert!(state.calibration_note.contains("checksum"));
    drop(state);
    let mut wrapper: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let mut payload: serde_json::Value =
        serde_json::from_str(wrapper["payload"].as_str().unwrap()).unwrap();
    payload["version"] = serde_json::json!(999);
    let text = payload.to_string();
    wrapper["sha256"] = serde_json::json!(sha(text.as_bytes()));
    wrapper["payload"] = serde_json::json!(text);
    std::fs::write(&p, wrapper.to_string()).unwrap();
    let state = State::open(&model, &cache, None, None).unwrap();
    assert_eq!(state.global(), None);
    assert!(state.calibration_note.contains("schema"));
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejected_model_cache_paths_leave_no_directories_or_source_changes() {
    use weight_atlas_rust::state::State;
    let (root, model, _) = small_model("cache-containment");
    let original = std::fs::read(model.join("model.safetensors")).unwrap();
    let alias = root.join("model-alias");
    std::os::unix::fs::symlink(&model, &alias).unwrap();
    for cache in [
        model.clone(),
        model.join("new-cache/nested"),
        alias.join("alias-cache/nested"),
        root.join("missing/../model/dot-cache"),
    ] {
        let error = State::open(&model, &cache, None, None).err().unwrap();
        assert!(error.to_string().contains("outside the read-only"));
        assert_eq!(std::fs::read_dir(&model).unwrap().count(), 1);
        assert_eq!(
            std::fs::read(model.join("model.safetensors")).unwrap(),
            original
        );
        assert!(!root.join("missing").exists());
    }
    // Normalize before mkdir, so even unused intermediate components stay absent.
    let state = State::open(
        &model,
        &model.join("never-created/../../outside-cache/nested"),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        state.root,
        root.join("outside-cache/nested").canonicalize().unwrap()
    );
    assert_eq!(std::fs::read_dir(&model).unwrap().count(), 1);
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn magnitude_cancellation_inspection_and_versioned_cache() {
    use weight_atlas_rust::{server, sha, state::State};
    let (root, model, cache) = small_model("magnitude");
    let state = State::open(&model, &cache, None, None).unwrap();
    assert!(state.tile(0, "tensor_magnitude", 0, 0, 0).is_err());
    state.calibrate_one(0).unwrap();
    let t = state.source.tensor(0).unwrap();
    let l = render::legend("tensor_magnitude", state.stats(0).as_ref(), None).unwrap();
    assert_eq!(l["min"], 0.);
    assert_eq!(l["max"], 1.);
    assert_eq!(l["scope"], "complete original tensor");
    let mag = render::mapping(
        "tensor_magnitude",
        &l,
        weight_atlas_rust::source::Dtype::Bf16,
    )
    .unwrap();
    let signed =
        render::mapping("tensor_linear", &l, weight_atlas_rust::source::Dtype::Bf16).unwrap();
    let (fields, metrics) =
        render::tile_fields(&state.source, t, &[&mag, &signed], 0, 0, 0).unwrap();
    assert_eq!(fields, [vec![1.], vec![0.]]);
    assert_eq!(
        (metrics.width, metrics.height, metrics.source_count),
        (1, 1, 2)
    );
    let q = server::query(&[
        ("tensor".into(), "0".into()),
        ("col".into(), "1".into()),
        ("left".into(), "tensor_magnitude".into()),
        ("right".into(), "tensor_linear".into()),
    ]);
    let scalar = server::inspect(&state, &q).unwrap();
    assert_eq!(scalar["raw_exact"], "-1");
    assert_eq!(scalar["bf16_hex_le"], "80bf");
    assert_eq!(scalar["native_indices"], serde_json::json!([1]));
    assert_eq!(scalar["transformed"]["left"], 1.);
    assert_eq!(scalar["transformed"]["right"], -1.);
    // A stale v1 payload at the otherwise identical key must never be consumed.
    let old_key = sha(format!(
        "{}:rust-rgba-v1-f64-row-sums-subfilter:0:tensor_magnitude:0:0:0:{l}",
        state.source.identity
    )
    .as_bytes());
    state
        .cache
        .lock()
        .unwrap()
        .put(&old_key, b"stale signed payload")
        .unwrap();
    let (png, cached, _) = state.tile(0, "tensor_magnitude", 0, 0, 0).unwrap();
    assert!(!cached);
    assert_eq!(
        png,
        render::png_for_rule("tensor_magnitude", &[1.], 1, 1).unwrap()
    );
    assert!(state.tile(0, "tensor_magnitude", 0, 0, 0).unwrap().1);
    assert_ne!(png, state.tile(0, "tensor_linear", 0, 0, 0).unwrap().0);
    drop(state);
    let state = State::open(&model, &cache, None, None).unwrap();
    assert_eq!(state.tile(0, "tensor_magnitude", 0, 0, 0).unwrap().0, png);
    assert!(state.tile(0, "tensor_magnitude", 0, 0, 0).unwrap().1);
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn magnitude_palette_zero_and_all_finite_bf16_words() {
    let mut s = Stats {
        count: 1,
        max_abs: value(0x7f7f),
        median_nonzero_abs: 1.,
        q99: 1.,
        robust_clipped_count: 0,
        histogram_sha256: Some("fixture".into()),
        exact_zero_count: 0,
        unique_bit_patterns: Some(1),
        dtype: "BF16".into(),
        calibration_method: "exact-16-bit-histogram".into(),
        seconds: 0.,
    };
    let l = render::legend("tensor_magnitude", Some(&s), None).unwrap();
    let lut = render::lookup("tensor_magnitude", &l);
    for bits in 0..=u16::MAX {
        let x = value(bits);
        if x.is_finite() {
            assert_eq!(lut[bits as usize], x.abs() / s.max_abs);
        }
    }
    s.max_abs = 0.;
    let zero = render::lookup(
        "tensor_magnitude",
        &render::legend("tensor_magnitude", Some(&s), None).unwrap(),
    );
    assert!(zero.iter().all(|&x| x == 0.));
    assert_eq!(
        render::rgba_for_rule("tensor_magnitude", &[0., 1.]),
        [247, 244, 249, 255, 84, 39, 143, 255]
    );
    for rule in [
        "global_linear",
        "global_asinh",
        "tensor_linear",
        "tensor_asinh",
    ] {
        assert_eq!(
            render::rgba_for_rule(rule, &[-1., 0., 1.]),
            render::rgba(&[-1., 0., 1.])
        );
        assert_eq!(
            render::png_for_rule(rule, &[-1., 0., 1.], 3, 1).unwrap(),
            render::png(&[-1., 0., 1.], 3, 1).unwrap()
        );
    }
}
