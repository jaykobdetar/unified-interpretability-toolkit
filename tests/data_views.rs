#[path = "support/workspace.rs"]
mod workspace;

use serde_json::json;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use weight_atlas_rust::{
    render,
    slice::{parse_indices, TensorSlice},
    source::{Dtype, Source},
    strength::StrengthProfile,
};

fn fixture(label: &str, shape: &[usize], values: &[f32], extras: bool) -> PathBuf {
    let root = std::env::temp_dir().join(format!("atlas-data-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut raw = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let mut header = json!({"weights":{"dtype":"F32","shape":shape,"data_offsets":[0,raw.len()]}});
    if extras {
        let offset = raw.len();
        raw.extend([1, 2, 3, 4]);
        header["packed"] = json!({"dtype":"I32","shape":[1],"data_offsets":[offset,offset+4]});
        header["empty"] = json!({"dtype":"F16","shape":[0],"data_offsets":[offset+4,offset+4]});
        header["scalar"] = json!({"dtype":"F32","shape":[],"data_offsets":[offset+4,offset+8]});
        raw.extend(2f32.to_le_bytes());
    }
    let header = header.to_string();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend(raw);
    std::fs::write(root.join("model.safetensors"), file).unwrap();
    root
}

#[test]
fn explicit_slices_preserve_native_indices_and_exact_bytes_in_mixed_catalog() {
    let values = (0..120).map(|i| i as f32 - 37.).collect::<Vec<_>>();
    let root = fixture("slices", &[2, 3, 4, 5], &values, true);
    let source = Source::open(&root).unwrap();
    assert_eq!(source.tensors.len(), 4);
    assert_eq!(source.tensors.iter().filter(|t| t.available).count(), 1);
    let t = source.tensors.iter().find(|t| t.name == "weights").unwrap();
    assert_eq!((t.rows, t.cols, t.count), (4, 5, 120));
    assert!(source.scalar_bits(t, 0, 0).is_err());
    for invalid in [&[][..], &[0][..], &[0, 3][..], &[2, 0][..], &[0, 0, 0][..]] {
        assert!(TensorSlice::new(&source, t.id, invalid).is_err());
    }
    for a in 0..2 {
        for b in 0..3 {
            let slice = TensorSlice::new(&source, t.id, &[a, b]).unwrap();
            assert_eq!(slice.element_start, (a * 3 + b) * 20);
            for row in 0..4 {
                for col in 0..5 {
                    let (bits, offset) = source.scalar_bits(&slice.tensor, row, col).unwrap();
                    let index = ((a * 3 + b) * 4 + row) * 5 + col;
                    assert_eq!(bits, values[index].to_bits());
                    assert_eq!(offset, t.byte_offset + 4 * index as u64);
                    assert_eq!(
                        slice.native_indices(row, col).unwrap(),
                        vec![a, b, row, col]
                    );
                }
            }
        }
    }
    assert_ne!(
        TensorSlice::new(&source, t.id, &[0, 0]).unwrap().identity,
        TensorSlice::new(&source, t.id, &[0, 1]).unwrap().identity
    );
    for s in ["-1", "1,", "1,,0", "+1", " 1", "1.0"] {
        assert!(parse_indices(s).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn display_limit_applies_to_trailing_axes_not_large_leading_dimensions() {
    for (label, shape, available) in [
        ("large-leading", [200001, 1, 1], true),
        ("large-row", [1, 200001, 1], false),
        ("large-column", [1, 1, 200001], false),
    ] {
        let root = std::env::temp_dir().join(format!("atlas-data-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let header =
            json!({"weights":{"dtype":"BF16","shape":shape,"data_offsets":[0,400002]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        let origin = bytes.len();
        bytes.resize(origin + 400002, 0);
        bytes[origin..origin + 2].copy_from_slice(&0x8000u16.to_le_bytes());
        bytes[origin + 400000..].copy_from_slice(&0x3f80u16.to_le_bytes());
        std::fs::write(root.join("model.safetensors"), bytes).unwrap();
        let source = Source::open(&root).unwrap();
        let t = &source.tensors[0];
        assert_eq!(t.available, available);
        if available {
            assert_eq!((t.rows, t.cols, t.count), (1, 1, 200001));
            let first = TensorSlice::new(&source, 0, &[0]).unwrap();
            let last = TensorSlice::new(&source, 0, &[200000]).unwrap();
            assert_eq!(
                source.scalar_bits(&first.tensor, 0, 0).unwrap(),
                (0x8000, t.byte_offset)
            );
            assert_eq!(
                source.scalar_bits(&last.tensor, 0, 0).unwrap(),
                (0x3f80, t.byte_offset + 400000)
            );
            assert_eq!(last.native_indices(0, 0).unwrap(), vec![200000, 0, 0]);
            assert!(TensorSlice::new(&source, 0, &[200001]).is_err());
        } else {
            assert!(t
                .unavailable_reason
                .as_deref()
                .unwrap()
                .contains("display-axis limit"));
            assert!(TensorSlice::new(&source, 0, &[0]).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn mixed_catalog_distinguishes_local_completion_from_unavailable_global_scope() {
    let _isolation = workspace::guard();
    use weight_atlas_rust::state::State;
    let root = fixture(
        "mixed-calibration-status",
        &[2, 2],
        &[1., -2., 3., -4.],
        true,
    );
    let cache = root.with_extension("cache");
    let state = State::open(&root, &cache, None, None).unwrap();
    let id = state
        .source
        .tensors
        .iter()
        .find(|t| t.name == "weights")
        .unwrap()
        .id;
    let before = state.model().unwrap();
    assert_eq!(before["global_calibration_supported"], false);
    assert!(before["global_calibration_unavailable_reason"]
        .as_str()
        .unwrap()
        .contains("unavailable tensors"));
    assert_eq!(before["supported_calibration_complete"], false);
    assert_eq!(before["coverage"]["supported_tensors"], 1);
    assert_eq!(before["coverage"]["unavailable_tensors"], 3);
    let status = state.status(Some(id)).unwrap();
    assert_eq!(status["model_status_version"], 1);
    assert_eq!(status["source_identity"], before["source_identity"]);
    assert_eq!(status["model_identity"], before["model_identity"]);
    assert_eq!(status["coverage"]["calibrated_tensors"], 0);
    assert_eq!(status["coverage"]["supported_tensors"], 1);
    assert_eq!(status["coverage"]["unavailable_tensors"], 3);
    assert_eq!(status["tensor_status"]["id"], id);
    assert!(status.get("catalog").is_none());
    assert!(serde_json::to_vec(&status).unwrap().len() <= 16384);
    for tensor in before["catalog"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["available"] == false)
    {
        assert!(tensor["rule_status"]["tensor_signed_percentile"]
            .as_str()
            .unwrap()
            .contains("unavailable"));
        let unavailable = state
            .status(Some(tensor["id"].as_u64().unwrap() as usize))
            .unwrap();
        assert!(
            unavailable["tensor_status"]["rule_status"]["tensor_signed_percentile"]
                .as_str()
                .unwrap()
                .contains("unavailable")
        );
    }
    state.calibrate_one(id).unwrap();
    let after = state.model().unwrap();
    assert_eq!(after["supported_calibration_complete"], true);
    assert_eq!(after["calibration_complete"], false);
    assert_eq!(after["coverage"]["statistics_complete"], false);
    let status = state.status(Some(id)).unwrap();
    assert_eq!(status["supported_calibration_complete"], true);
    assert_eq!(status["global_calibration_supported"], false);
    assert_eq!(status["coverage"]["calibrated_tensors"], 1);
    assert_eq!(status["tensor_status"]["calibration_complete"], true);
    assert!(state.global().is_none());
    assert!(state
        .legends(
            state.source.tensor(id).unwrap(),
            "tensor_linear",
            "tensor_magnitude"
        )
        .is_ok());
    assert!(state
        .legends(
            state.source.tensor(id).unwrap(),
            "global_linear",
            "tensor_magnitude"
        )
        .is_err());
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
fn external_profile_deadline_exhausts_grant_without_reading_or_reauthorizing() {
    let root = fixture("external-deadline", &[3, 4], &[1.; 12], false);
    let source = Source::open(&root).unwrap();
    let slice = TensorSlice::new(&source, 0, &[]).unwrap();
    let mut p = StrengthProfile::new(
        &source,
        slice,
        &"a".repeat(64),
        42,
        12,
        Duration::from_secs(1),
    )
    .unwrap();
    p.advance_until(&source, 2, Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(p.progress()["visited_values"], 2);
    let past = Instant::now() - Duration::from_millis(1);
    p.advance_until(&source, 2, past).unwrap();
    assert_eq!(p.progress()["visited_values"], 2);
    assert_eq!(p.progress()["state"], "paused");
    p.advance_until(&source, 12, Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(p.progress()["visited_values"], 2);
    // Only an explicit new authorization restores a value/time allowance.
    p.authorize(10, Duration::from_secs(1)).unwrap();
    p.advance_until(&source, 1, Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(p.progress()["visited_values"], 3);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn profiles_resume_only_within_authorization_and_match_direct_means() {
    let values = [-3., 0., 2., 9., -1., 4., 5., -8., -0., 7., 6., -2.];
    let root = fixture("strength", &[3, 4], &values, false);
    let source = Source::open(&root).unwrap();
    let slice = TensorSlice::new(&source, 0, &[]).unwrap();
    let mut p = StrengthProfile::new(
        &source,
        slice,
        &"a".repeat(64),
        42,
        5,
        Duration::from_secs(1),
    )
    .unwrap();
    p.advance(&source).unwrap();
    assert_eq!(p.progress()["visited_values"], 5);
    assert_eq!(p.progress()["state"], "paused");
    p.advance(&source).unwrap();
    assert_eq!(p.progress()["visited_values"], 5);
    let first = p.page(&source, "rows", 0, 3).unwrap();
    assert_eq!(first["original"][0]["complete"], true);
    assert_eq!(first["original"][1]["visited_count"], 1);
    assert_eq!(first["original"][2]["mean_abs"], serde_json::Value::Null);
    p.authorize(7, Duration::from_secs(1)).unwrap();
    p.advance(&source).unwrap();
    assert_eq!(p.progress()["complete"], true);
    for (axis, n, expected) in [("rows", 3, 4), ("columns", 4, 3)] {
        let page = p.page(&source, axis, 0, n).unwrap();
        for i in 0..n {
            let sum: f64 = values
                .iter()
                .enumerate()
                .filter(|(j, _)| {
                    if axis == "rows" {
                        j / 4 == i
                    } else {
                        j % 4 == i
                    }
                })
                .map(|(_, v)| v.abs() as f64)
                .sum();
            assert_eq!(page["original"][i]["mean_abs"], sum / expected as f64);
            assert_eq!(page["control"][i]["visited_count"], expected);
            assert_eq!(page["control"][i]["complete"], true);
        }
        let total: f64 = page["control"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["sum_abs"].as_f64().unwrap())
            .sum();
        assert_eq!(total, values.iter().map(|v| v.abs() as f64).sum::<f64>());
    }
    assert!(p.page(&source, "rows", 0, 1025).is_err());
    assert!(p.authorize(1, Duration::from_secs(1)).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn nonfinite_and_source_mutation_invalidate_profiles() {
    let root = fixture("nonfinite", &[1, 2], &[1., f32::INFINITY], false);
    let source = Source::open(&root).unwrap();
    let mut p = StrengthProfile::new(
        &source,
        TensorSlice::new(&source, 0, &[]).unwrap(),
        &"b".repeat(64),
        0,
        2,
        Duration::from_secs(1),
    )
    .unwrap();
    assert!(p.advance(&source).is_err());
    assert_eq!(p.progress()["state"], "invalid");
    assert!(p.page(&source, "rows", 0, 1).is_err());
    assert!(p.authorize(1, Duration::from_secs(1)).is_err());
    std::fs::remove_dir_all(root).unwrap();
    let root = fixture("mutation", &[1, 2], &[1., 2.], false);
    let source = Source::open(&root).unwrap();
    let mut p = StrengthProfile::new(
        &source,
        TensorSlice::new(&source, 0, &[]).unwrap(),
        &"c".repeat(64),
        0,
        2,
        Duration::from_secs(1),
    )
    .unwrap();
    std::fs::write(root.join("model.safetensors"), b"changed").unwrap();
    assert!(p.advance(&source).is_err());
    assert_eq!(p.progress()["state"], "invalid");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn typical_magnitude_handles_ties_zero_fallback_and_pointwise_pooling() {
    let _isolation = workspace::guard();
    for (label, values) in [
        ("zeros", vec![0., -0.]),
        ("sparse", {
            let mut v = vec![0.; 1001];
            v[1000] = 2.;
            v
        }),
        ("ties", vec![-2., 2., 0., 10.]),
    ] {
        let root = fixture(label, &[1, values.len()], &values, false);
        let source = Source::open(&root).unwrap();
        let t = &source.tensors[0];
        let (stats, _) = render::calibrate(&source, t).unwrap();
        let legend = render::legend("tensor_magnitude_asinh", Some(&stats), None).unwrap();
        assert_eq!(legend["min"], 0.);
        assert_eq!(legend["palette"], "sequential-purple-v1");
        let s = legend["s"].as_f64().unwrap();
        let d = legend["max"].as_f64().unwrap();
        let lut = render::mapping("tensor_magnitude_asinh", &legend, Dtype::F32).unwrap();
        for value in &values {
            let expected = ((value.abs() as f64 / s).asinh() / (d / s).asinh()).min(1.);
            assert!((lut.value(value.to_bits()) - expected).abs() < 1e-15);
        }
        let (fields, _) = render::region_fields(
            &source,
            t,
            &[&lut],
            (0, 1, 0, values.len()),
            values.len().next_power_of_two(),
        )
        .unwrap();
        let expected =
            values.iter().map(|v| lut.value(v.to_bits())).sum::<f64>() / values.len() as f64;
        assert!((fields[0][0] - expected).abs() < 1e-14);
        if label == "zeros" {
            assert_eq!(s, 1.);
            assert_eq!(d, 1.);
            assert_eq!(fields[0][0], 0.);
        }
        if label == "sparse" {
            assert_eq!(legend["clipped_count"], 1);
            assert_eq!(legend["zero_quantile_fallback"], true);
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn narrow_batches_respect_stride_edges_and_do_not_decode_gaps() {
    let _isolation = workspace::guard();
    let mut values = (0..60).map(|v| v as f32).collect::<Vec<_>>();
    // These gap columns are read as bytes but must never enter the field.
    for row in 0..6 {
        values[row * 10] = f32::NAN;
        values[row * 10 + 9] = f32::INFINITY;
    }
    let root = fixture("narrow", &[6, 10], &values, false);
    let source = Source::open(&root).unwrap();
    let lut = render::mapping("tensor_linear", &json!({"max":60.,"s":null}), Dtype::F32).unwrap();
    let (fields, m) =
        render::region_fields(&source, &source.tensors[0], &[&lut], (0, 6, 2, 9), 2).unwrap();
    assert!(m.max_raw_band_values * 4 <= 2 * 1024 * 1024);
    assert_eq!(m.source_count, 42);
    assert_eq!(m.source_bytes_read, 57 * 4); // 6 narrow rows with native stride 10.
    for r in 0..3 {
        for c in 0..4 {
            let mut expected = Vec::new();
            for row in r * 2..(r + 1) * 2 {
                for col in 2 + c * 2..(4 + c * 2).min(9) {
                    expected.push(values[row * 10 + col] as f64 / 60.);
                }
            }
            assert!(
                (fields[0][r * 4 + c] - expected.iter().sum::<f64>() / expected.len() as f64).abs()
                    < 1e-15
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn full_profiles_cross_the_old_window_limit_and_expired_grants_do_no_work() {
    let values = vec![2.; 257 * 257];
    let root = fixture("full", &[257, 257], &values, false);
    let source = Source::open(&root).unwrap();
    let mut p = StrengthProfile::new(
        &source,
        TensorSlice::new(&source, 0, &[]).unwrap(),
        &"d".repeat(64),
        7,
        values.len(),
        Duration::from_nanos(1),
    )
    .unwrap();
    p.advance(&source).unwrap();
    assert_eq!(p.progress()["visited_values"], 0);
    assert_eq!(p.progress()["state"], "paused");
    p.advance(&source).unwrap();
    assert_eq!(p.progress()["visited_values"], 0);
    p.authorize(values.len(), Duration::from_secs(5)).unwrap();
    // Explicit bounded test driver; production scheduler must not reauthorize.
    for _ in 0..100 {
        if p.progress()["state"] != "ready" {
            break;
        }
        p.advance(&source).unwrap();
    }
    assert_eq!(p.progress()["complete"], true);
    let page = p.page(&source, "rows", 0, 257).unwrap();
    for side in ["original", "control"] {
        for row in page[side].as_array().unwrap() {
            assert_eq!(row["mean_abs"], 2.);
            assert_eq!(row["visited_count"], 257);
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn profile_snapshots_restore_exhausted_and_preserve_split_accumulation() {
    use std::io::Cursor;
    use weight_atlas_rust::{sha, strength::snapshot};
    let root = fixture(
        "snapshot",
        &[3, 4],
        &[1., -3., 0., 2., 4., -9., 0.5, 1., 7., 8., 2., -4.],
        false,
    );
    let source = Source::open(&root).unwrap();
    let model = "a".repeat(64);
    let make = |values| {
        StrengthProfile::new(
            &source,
            TensorSlice::new(&source, 0, &[]).unwrap(),
            &model,
            17,
            values,
            Duration::from_secs(1),
        )
        .unwrap()
    };
    let mut partial = make(5);
    partial.advance(&source).unwrap();
    let mut bytes = Vec::new();
    let revision = partial
        .write_snapshot(&source, &mut bytes, Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(sha(&bytes), revision);
    let restore = |raw: &[u8], expected: &str| {
        StrengthProfile::restore_snapshot(
            &source,
            TensorSlice::new(&source, 0, &[]).unwrap(),
            &model,
            17,
            Cursor::new(raw),
            raw.len(),
            expected,
            Instant::now() + Duration::from_secs(1),
        )
    };
    let mut restored = restore(&bytes, &revision).unwrap();
    assert_eq!(restored.progress()["state"], "paused");
    assert_eq!(restored.progress()["authorized_remaining_values"], 0);
    restored.advance(&source).unwrap();
    assert_eq!(restored.visited_values(), 5);
    restored.authorize(7, Duration::from_secs(1)).unwrap();
    restored.advance(&source).unwrap();
    let mut complete = make(12);
    complete.advance(&source).unwrap();
    let mut a = Vec::new();
    let mut b = Vec::new();
    restored
        .write_snapshot(&source, &mut a, Instant::now() + Duration::from_secs(1))
        .unwrap();
    complete
        .write_snapshot(&source, &mut b, Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(a, b); // Bit-preserving compensation and exact counts.
    assert!(restore(&bytes, &"0".repeat(64)).is_err());
    assert!(restore(&bytes[..bytes.len() - 1], &revision).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(restore(&trailing, &sha(&trailing)).is_err());
    let binding_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    for offset in [
        8,
        16,
        40,
        48,
        72,
        104,
        136,
        snapshot::HEADER_BYTES + binding_len + 16,
    ] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 1;
        assert!(
            restore(&corrupt, &sha(&corrupt)).is_err(),
            "offset {offset}"
        );
    }
    let mut corrupt = bytes.clone();
    let start = snapshot::HEADER_BYTES + binding_len;
    corrupt[start..start + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(restore(&corrupt, &sha(&corrupt)).is_err());
    assert!(snapshot::layout(200000, 200000, 1024).is_err());
    assert!(snapshot::layout(3, 4, 1024).unwrap().live_bytes <= 32 * 1024 * 1024);
    assert!(StrengthProfile::restore_snapshot(
        &source,
        TensorSlice::new(&source, 0, &[]).unwrap(),
        &model,
        17,
        Cursor::new(&bytes),
        bytes.len(),
        &revision,
        Instant::now()
    )
    .is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn profile_snapshot_sums_obey_each_exact_source_dtype_maximum() {
    use std::io::Cursor;
    use weight_atlas_rust::{sha, strength::snapshot};
    for (dtype, raw, maximum) in [
        ("F16", 0x7bffu16.to_le_bytes().to_vec(), 65504.0),
        (
            "BF16",
            0x7f7fu16.to_le_bytes().to_vec(),
            f32::from_bits(0x7f7f0000) as f64,
        ),
        ("F32", f32::MAX.to_le_bytes().to_vec(), f32::MAX as f64),
    ] {
        let root = std::env::temp_dir().join(format!(
            "atlas-snapshot-dtype-{dtype}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let header = json!({"weights":{"dtype":dtype,"shape":[1,1],
            "data_offsets":[0,raw.len()]}})
        .to_string();
        let mut payload = (header.len() as u64).to_le_bytes().to_vec();
        payload.extend(header.as_bytes());
        payload.extend(raw);
        std::fs::write(root.join("model.safetensors"), payload).unwrap();
        let source = Source::open(&root).unwrap();
        let model = "a".repeat(64);
        let mut profile = StrengthProfile::new(
            &source,
            TensorSlice::new(&source, 0, &[]).unwrap(),
            &model,
            17,
            1,
            Duration::from_secs(1),
        )
        .unwrap();
        profile.advance(&source).unwrap();
        let mut bytes = Vec::new();
        let revision = profile
            .write_snapshot(&source, &mut bytes, Instant::now() + Duration::from_secs(1))
            .unwrap();
        let restore = |encoded: &[u8], digest: &str| {
            StrengthProfile::restore_snapshot(
                &source,
                TensorSlice::new(&source, 0, &[]).unwrap(),
                &model,
                17,
                Cursor::new(encoded),
                encoded.len(),
                digest,
                Instant::now() + Duration::from_secs(1),
            )
        };
        assert!(restore(&bytes, &revision).is_ok(), "{dtype} exact maximum");
        let binding_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let start = snapshot::HEADER_BYTES + binding_len;
        assert_eq!(
            f64::from_le_bytes(bytes[start..start + 8].try_into().unwrap()),
            maximum
        );
        // Preserve paired sums, counts, compensation and checksum so that only
        // the source dtype's finite ceiling distinguishes this invalid frame.
        for axis in 0..4 {
            let offset = start + axis * 24;
            bytes[offset..offset + 8].copy_from_slice(&(maximum * 1.000001).to_le_bytes());
        }
        assert!(
            restore(&bytes, &sha(&bytes)).is_err(),
            "{dtype} above maximum"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
