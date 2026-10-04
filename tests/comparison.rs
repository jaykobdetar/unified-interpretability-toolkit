use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use weight_atlas_rust::{comparison::Comparison, source::Dtype};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "atlas-compare-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
fn model(root: &Path, name: &str, entries: &[(&str, Dtype, Vec<usize>, Vec<u32>)]) -> PathBuf {
    let path = root.join(name);
    std::fs::create_dir_all(&path).unwrap();
    let mut header = serde_json::Map::new();
    let mut raw = Vec::new();
    for (name, dtype, shape, bits) in entries {
        let start = raw.len();
        for b in bits {
            raw.extend_from_slice(&b.to_le_bytes()[..dtype.bytes()]);
        }
        header.insert(
            (*name).into(),
            json!({"dtype":dtype.name(),"shape":shape,"data_offsets":[start,raw.len()]}),
        );
    }
    let text = serde_json::to_vec(&header).unwrap();
    let mut all = (text.len() as u64).to_le_bytes().to_vec();
    all.extend(text);
    all.extend(raw);
    std::fs::write(path.join("model.safetensors"), all).unwrap();
    path
}
fn f32s(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
#[test]
fn mixed_sources_are_exact_while_difference_is_labeled_derived() {
    let r = root("mixed");
    let a = model(
        &r,
        "a",
        &[(
            "w",
            Dtype::Bf16,
            vec![2, 3],
            vec![0xbf80, 0, 0x8000, 0x3f80, 0x4000, 0x4040],
        )],
    );
    let b = model(
        &r,
        "b",
        &[(
            "w",
            Dtype::F32,
            vec![2, 3],
            f32s(&[1., 0., -0., 1., 4., -3.]),
        )],
    );
    let c = Comparison::open(&a, &b, &r.join("cache")).unwrap();
    let raw = c.inspect(0, 0, 0).unwrap();
    assert_eq!(raw["originals"]["a"]["raw_exact"], "-1");
    assert_eq!(raw["originals"]["b"]["raw_exact"], "1");
    assert_eq!(raw["difference"]["value"], 2.);
    assert_eq!(raw["inference_editable"], false);
    assert!(raw.get("tensor").is_none());
    assert!(raw["difference"]["derived"].as_bool().unwrap());
    assert_eq!(
        c.inspect(0, 0, 2).unwrap()["originals"]["a"]["raw_exact"],
        "-0.0"
    );
    assert!(c.view(0, "a", "b", "linear").is_err());
    c.calibrate_one(0).unwrap();
    let s = c.scales(0).unwrap();
    assert_eq!(s.shared_raw_max, 4.);
    assert_eq!(s.difference_max, 6.);
    assert_eq!(s.nonzero_difference_count, 3);
    let v = c.view(0, "a", "b", "linear").unwrap();
    assert_eq!(v["legends"]["left"]["max"], v["legends"]["right"]["max"]);
    assert_eq!(
        c.legend(0, "delta", "linear").unwrap()["calibration_domain"],
        "difference"
    );
    let (a, _) = c.fields(0, "a", "linear", 2, 0, 0).unwrap();
    let (b, _) = c.fields(0, "b", "linear", 2, 0, 0).unwrap();
    let (d, m) = c.fields(0, "delta", "linear", 2, 0, 0).unwrap();
    assert_eq!(a[0], -0.25);
    assert_eq!(b[0], 0.25);
    assert_eq!(d[0], 1. / 3.);
    assert_eq!(m.source_bytes_read, 36);
    assert!(m.max_raw_band_values / 2 * 6 <= 2 * 1024 * 1024);
    drop(c);
    std::fs::remove_dir_all(r).unwrap();
}
#[test]
fn transform_precedes_pooling_and_partial_edges_keep_native_counts() {
    let r = root("pool");
    let a = model(&r, "a", &[("w", Dtype::F16, vec![3, 5], vec![0; 15])]);
    let vals = vec![
        -2., 2., -2., 2., 1., 2., -2., 2., -2., 3., 1., 1., 1., 1., 4.,
    ];
    let b = model(&r, "b", &[("w", Dtype::F32, vec![3, 5], f32s(&vals))]);
    let c = Comparison::open(&a, &b, &r.join("cache")).unwrap();
    c.calibrate_one(0).unwrap();
    let (signed, m) = c.fields(0, "delta", "linear", 2, 0, 0).unwrap();
    let (absolute, _) = c.fields(0, "abs_delta", "linear", 2, 0, 0).unwrap();
    let (magnitude, _) = c.fields(0, "delta", "magnitude", 2, 0, 0).unwrap();
    assert_eq!((m.width, m.height, m.factor), (3, 2, 2));
    assert_eq!(signed, vec![0., 0., 0.5, 0.25, 0.25, 1.]);
    assert_eq!(absolute, vec![0.5, 0.5, 0.5, 0.25, 0.25, 1.]);
    assert_eq!(absolute, magnitude);
    for mapping in ["linear", "asinh", "magnitude"] {
        let (f, _) = c.fields(0, "a", mapping, 0, 0, 0).unwrap();
        assert_eq!(f, vec![0.]);
    }
    drop(c);
    std::fs::remove_dir_all(r).unwrap();
}
#[test]
fn complete_name_shape_and_dtype_policy_is_checked_before_cache_creation() {
    let r = root("mismatch");
    let a = model(&r, "a", &[("w", Dtype::F32, vec![2], f32s(&[1., 2.]))]);
    for (label, name, shape) in [("name", "other", vec![2]), ("shape", "w", vec![1, 2])] {
        let b = model(&r, label, &[(name, Dtype::F32, shape, f32s(&[1., 2.]))]);
        let cache = r.join(format!("cache-{label}"));
        let e = Comparison::open(&a, &b, &cache).err().unwrap().to_string();
        assert!(e.contains("Complete named-shape compatibility failed"));
        assert!(!cache.exists());
    }
    let b = model(
        &r,
        "extra",
        &[
            ("w", Dtype::F32, vec![2], f32s(&[1., 2.])),
            ("extra", Dtype::F16, vec![1], vec![0]),
        ],
    );
    assert!(Comparison::open(&a, &b, &r.join("cache-extra")).is_err());
    let bad = r.join("bad");
    std::fs::create_dir_all(&bad).unwrap();
    let header = br#"{"w":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header);
    data.extend([0; 4]);
    std::fs::write(bad.join("model.safetensors"), data).unwrap();
    assert!(Comparison::open(&a, &bad, &r.join("cache-bad")).is_err());
    std::fs::remove_dir_all(r).unwrap();
}
#[test]
fn nonfinite_inspection_is_preserved_but_no_calibration_or_tile_is_published() {
    let r = root("nonfinite");
    let a = model(
        &r,
        "a",
        &[("w", Dtype::F32, vec![2], f32s(&[0., f32::NAN]))],
    );
    let b = model(&r, "b", &[("w", Dtype::F16, vec![2], vec![0, 0x7c00])]);
    let c = Comparison::open(&a, &b, &r.join("cache")).unwrap();
    let v = c.inspect(0, 0, 1).unwrap();
    assert_eq!(v["originals"]["a"]["classification"], "nan");
    assert_eq!(v["originals"]["b"]["classification"], "infinity");
    assert!(v["difference"]["value"].is_null());
    assert!(c.calibrate_one(0).is_err());
    assert!(c.scales(0).is_none());
    assert!(!r.join("cache/comparison-calibration.json").exists());
    assert!(c.tile(0, "delta", "linear", 0, 0, 0).is_err());
    drop(c);
    std::fs::remove_dir_all(r).unwrap();
}
#[test]
fn cache_direction_identity_and_missing_tiles_are_segregated() {
    let r = root("cache");
    let a = model(&r, "a", &[("w", Dtype::F32, vec![2], f32s(&[1., 2.]))]);
    let b = model(&r, "b", &[("w", Dtype::F32, vec![2], f32s(&[2., 4.]))]);
    let cache = r.join("cache");
    let c = Comparison::open(&a, &b, &cache).unwrap();
    let identity = c.identity.clone();
    c.calibrate_one(0).unwrap();
    let first = c.tile(0, "delta", "linear", 1, 0, 0).unwrap();
    assert!(!first.1);
    assert!(c.tile(0, "delta", "linear", 1, 0, 0).unwrap().1);
    assert_ne!(first.0, c.tile(0, "a", "linear", 1, 0, 0).unwrap().0);
    for entry in std::fs::read_dir(cache.join("tiles")).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    assert!(!c.tile(0, "delta", "linear", 1, 0, 0).unwrap().1);
    drop(c);
    let c = Comparison::open(&a, &b, &cache).unwrap();
    assert!(c.scales(0).is_some());
    assert!(c.tile(0, "delta", "linear", 1, 0, 0).unwrap().1);
    drop(c);
    let reversed = Comparison::open(&b, &a, &cache).unwrap();
    assert_ne!(reversed.identity, identity);
    assert!(reversed.scales(0).is_none());
    reversed.calibrate_one(0).unwrap();
    let (field, _) = reversed.fields(0, "delta", "linear", 1, 0, 0).unwrap();
    assert_eq!(field, vec![-0.5, -1.]);
    assert!(!reversed.tile(0, "delta", "linear", 1, 0, 0).unwrap().1);
    drop(reversed);
    std::fs::remove_dir_all(r).unwrap();
}
#[test]
fn source_mutation_and_outputs_inside_either_source_fail_closed() {
    let r = root("boundary");
    let a = model(&r, "a", &[("w", Dtype::Bf16, vec![2], vec![0, 0])]);
    let b = model(&r, "b", &[("w", Dtype::Bf16, vec![2], vec![0, 0])]);
    assert!(Comparison::open(&a, &b, &b.join("cache")).is_err());
    assert!(!b.join("cache").exists());
    let c = Comparison::open(&a, &b, &r.join("cache")).unwrap();
    assert!(c.output_prefix(&a.join("output")).is_err());
    assert!(c.output_prefix(&b.join("missing/../output")).is_err());
    c.calibrate_one(0).unwrap();
    for q in ["a", "b", "delta", "abs_delta"] {
        for m in ["linear", "asinh", "magnitude"] {
            assert_eq!(c.fields(0, q, m, 0, 0, 0).unwrap().0, vec![0.]);
        }
    }
    assert!(c.legend(0, "delta", "robust99").is_err());
    assert!(c.legend(0, "delta", "signed_percentile").is_err());
    assert!(c.fields(0, "delta", "linear", 0, usize::MAX, 0).is_err());
    let p = b.join("model.safetensors");
    let mut bytes = std::fs::read(&p).unwrap();
    let last = bytes.len() - 1;
    bytes[last] = 0x3f;
    std::fs::write(p, bytes).unwrap();
    assert!(c.model().is_err());
    assert!(c.inspect(0, 0, 0).is_err());
    assert!(c.tile(0, "delta", "linear", 0, 0, 0).is_err());
    drop(c);
    std::fs::remove_dir_all(r).unwrap();
}
