//! Fixed compatibility vectors for later identity and response refactoring.
//! These inert catalogs isolate slice derivation, not filesystem fingerprints.
use std::path::PathBuf;
use weight_atlas_rust::{
    slice::TensorSlice,
    source::{Dtype, Source, Tensor},
};

fn catalog(name: &str, shape: &[usize], dtype: Dtype) -> Source {
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    Source {
        root: PathBuf::from("identity-vector"),
        tensors: vec![Tensor {
            id: 0,
            name: name.into(),
            shape: shape.to_vec(),
            rows,
            cols,
            count: shape.iter().product(),
            dtype: dtype.name().into(),
            element_bytes: dtype.bytes(),
            available: true,
            unavailable_reason: None,
            shard: "vector.safetensors".into(),
            shard_id: 0,
            byte_offset: 128,
            max_level: 3,
            min_level: 0,
        }],
        shards: Vec::new(),
        files: Vec::new(),
        identity: "a".repeat(64),
        bytes: 0,
        header_bytes: 0,
        index_fingerprint: None,
    }
}

#[test]
fn trailing_slice_identity_matches_fixed_utf8_vectors() {
    // Digests independently computed from the compact UTF-8 JSON arrays.
    // Expected digests are literals, never obtained from a second production call.
    for (name, shape, dtype, leading, expected) in [
        (
            "model.layers.0.attn.weight",
            vec![2, 3],
            Dtype::Bf16,
            vec![],
            "f363649376a6725433d8e741c5e588ea35476030085a0bb71ac1f12109bcafb2",
        ),
        (
            "model.layers.0.attn.weight",
            vec![2, 3, 4, 5],
            Dtype::Bf16,
            vec![1, 2],
            "ef787dcf800d162d27d12fb3d1b07b605d5370bb746f3dacb21de18fcbaf90e9",
        ),
        (
            "unicode.λ",
            vec![2, 3, 4, 5],
            Dtype::F32,
            vec![0, 1],
            "6ce35d119ae9d4fd9c21e297e54db8be5d1ce8cb23ab4d1bf8cc13b11ee0f3a2",
        ),
    ] {
        let source = catalog(name, &shape, dtype);
        let slice = TensorSlice::new(&source, 0, &leading).unwrap();
        assert_eq!(slice.identity, expected, "{name} {shape:?} {dtype:?}");
    }
}

#[test]
fn slice_binding_preserves_fixed_json_field_order_and_utf8_bytes() {
    let source = catalog("unicode.λ", &[2, 3, 4, 5], Dtype::F32);
    let slice = TensorSlice::new(&source, 0, &[0, 1]).unwrap();
    assert_eq!(
        serde_json::to_string(&slice.binding("model-vector")).unwrap(),
        concat!(
            r#"{"cols":5,"dtype":"F32","model_identity":"model-vector","name":"unicode.λ","rows":4,"shape":[2,3,4,5],"#,
            r#""slice":{"display_axes":[2,3],"leading_indices":[0,1]},"#,
            r#""source_identity":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tensor":0,"version":2}"#
        )
    );
}

#[test]
fn slice_descriptor_preserves_fixed_json_bytes_and_native_offsets() {
    let source = catalog("unicode.λ", &[2, 3, 4, 5], Dtype::F32);
    let slice = TensorSlice::new(&source, 0, &[0, 1]).unwrap();
    assert_eq!(
        serde_json::to_string(&slice.descriptor()).unwrap(),
        concat!(
            r#"{"byte_offset":208,"count":20,"display_axes":[2,3],"element_start":20,"#,
            r#""identity":"6ce35d119ae9d4fd9c21e297e54db8be5d1ce8cb23ab4d1bf8cc13b11ee0f3a2","leading_indices":[0,1],"#,
            r#""native_shape":[2,3,4,5],"version":1}"#
        )
    );
}
