//! Header-only source catalog and identity-envelope compatibility witnesses.
#[path = "support/workspace.rs"]
mod workspace;

use serde_json::json;
use sha2::{Digest, Sha256};
use std::{os::unix::fs::MetadataExt, path::PathBuf};
use weight_atlas_rust::source::Source;

struct Fixture {
    root: PathBuf,
    model: PathBuf,
    header: Vec<u8>,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "atlas-source-vectors-{label}-{}",
            std::process::id()
        ));
        let model = root.join("model-λ");
        std::fs::create_dir_all(&model).unwrap();
        let header = serde_json::to_vec(&json!({
            "__metadata__":{"format":"pt"},
            "a-matrix":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},
            "b-half":{"dtype":"F16","shape":[2],"data_offsets":[12,16]},
            "c-single":{"dtype":"F32","shape":[1,2],"data_offsets":[16,24]},
            "d-integer":{"dtype":"I8","shape":[3],"data_offsets":[24,27]},
            "e-unknown":{"dtype":"NEW_CODEC","shape":[3],"data_offsets":[27,30]},
            "f-scalar":{"dtype":"BF16","shape":[],"data_offsets":[30,32]},
            "g-empty":{"dtype":"BF16","shape":[0],"data_offsets":[32,32]}
        }))
        .unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&[0; 32]);
        std::fs::write(model.join("model.safetensors"), file).unwrap();
        Self {
            root,
            model,
            header,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn source_open_catalog_preserves_order_geometry_and_catalog_only_encodings() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("catalog");
    let result = Source::open(&fixture.model);
    assert!(result.is_ok(), "valid mixed catalog");
    let source = result.unwrap();
    assert_eq!(
        source
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        [
            "a-matrix",
            "b-half",
            "c-single",
            "d-integer",
            "e-unknown",
            "f-scalar",
            "g-empty"
        ],
        "catalog order"
    );
    for (id, tensor) in source.tensors.iter().enumerate() {
        assert_eq!(tensor.id, id, "catalog ID");
        assert_eq!(tensor.shard, "model.safetensors");
        assert_eq!(tensor.shard_id, 0);
    }
    assert_eq!(
        (
            source.tensors[0].rows,
            source.tensors[0].cols,
            source.tensors[0].count
        ),
        (2, 3, 6),
        "native geometry"
    );
    assert_eq!(
        (
            source.tensors[1].rows,
            source.tensors[1].cols,
            source.tensors[1].count
        ),
        (1, 2, 2)
    );
    assert_eq!(
        (
            source.tensors[2].rows,
            source.tensors[2].cols,
            source.tensors[2].count
        ),
        (1, 2, 2)
    );
    for (id, dtype, bytes) in [(0, "BF16", 2), (1, "F16", 2), (2, "F32", 4)] {
        assert_eq!(source.tensors[id].dtype, dtype);
        assert_eq!(source.tensors[id].element_bytes, bytes);
        assert!(source.tensors[id].available);
        assert_eq!(source.tensors[id].unavailable_reason, None);
    }
    for (id, dtype, bytes, reason) in [
        (
            3,
            "I8",
            1,
            "Storage extent validated; numeric encoding/quantization scale semantics unsupported",
        ),
        (
            4,
            "NEW_CODEC",
            0,
            "Unknown storage encoding; declared extent checked only, no numeric interpretation",
        ),
        (5, "BF16", 2, "Scalar/empty tensor display unsupported"),
        (6, "BF16", 2, "Scalar/empty tensor display unsupported"),
    ] {
        assert_eq!(source.tensors[id].dtype, dtype);
        assert_eq!(source.tensors[id].element_bytes, bytes);
        assert!(!source.tensors[id].available);
        assert_eq!(
            source.tensors[id].unavailable_reason.as_deref(),
            Some(reason)
        );
    }
    assert_eq!(source.header_bytes, fixture.header.len() as u64 + 8);
    assert_eq!(source.bytes, source.header_bytes + 32);
    assert!(source.check().is_ok());
}

#[test]
fn source_open_identity_preserves_independent_file_envelope() {
    let _isolation = workspace::guard();
    let fixture = Fixture::new("identity");
    let metadata = std::fs::metadata(fixture.model.join("model.safetensors")).unwrap();
    // These inputs come directly from the fixture and OS, not Source's records.
    let envelope = json!({
        "schema":1,
        "root":fixture.model.canonicalize().unwrap(),
        "index":null,
        "index_stat":null,
        "shards":[{
            "name":"model.safetensors",
            "header_sha256":format!("{:x}",Sha256::digest(&fixture.header)),
            "data_start":fixture.header.len() as u64+8,
            "fingerprint":{
                "size":metadata.len(),"dev":metadata.dev(),"inode":metadata.ino(),
                "mtime":metadata.mtime(),"mtime_ns":metadata.mtime_nsec(),
                "ctime":metadata.ctime(),"ctime_ns":metadata.ctime_nsec()
            }
        }]
    });
    let expected = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&envelope).unwrap())
    );
    let source = Source::open(&fixture.model).unwrap();
    assert_eq!(source.identity, expected, "source identity envelope");
}
