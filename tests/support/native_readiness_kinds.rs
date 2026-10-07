//! Preserve native readiness provenance and existing external error contracts.
use super::{Error, Readiness};
use crate::{comparison::Comparison, http_status::ReadinessStatus, render, state::State};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn hold(error: Error, reason: Readiness, message: &str) {
    assert!(
        matches!(&error, Error::Refusal(_)),
        "readiness refusal family"
    );
    assert_eq!(
        error.readiness(),
        Some(reason),
        "readiness typed provenance"
    );
    assert_eq!(error.to_string(), message, "readiness original display");
    assert_eq!(
        format!("{error:?}"),
        format!("{message:?}"),
        "readiness original debug"
    );
    assert!(error.source().is_none(), "readiness original source chain");
    assert_eq!(
        ReadinessStatus::from_native_error(&error).code(),
        503,
        "readiness contextual status"
    );
}

#[test]
fn explicit_reasons_preserve_all_original_error_envelopes() {
    for (reason, message) in [
        (
            Readiness::Checkpoint,
            "Full checkpoint calibration is not ready",
        ),
        (
            Readiness::Tensor,
            "Complete selected-tensor calibration is not ready",
        ),
        (
            Readiness::Histogram,
            "Exact percentile histogram is not ready",
        ),
        (
            Readiness::Pair,
            "Complete paired-tensor calibration is not ready",
        ),
    ] {
        hold(reason.error(), reason, message);
    }
}

#[test]
fn untagged_owned_and_standard_errors_keep_the_contextual_fallback() {
    for (message, code) in [
        ("", 400),
        ("Unknown tensor", 400),
        ("not ready", 503),
        ("雪 not ready λ", 503),
        ("NOT READY", 400),
    ] {
        for error in [Error::from(message), std::io::Error::other(message).into()] {
            assert_eq!(
                error.readiness(),
                None,
                "incidental text has no declared readiness reason"
            );
            assert_eq!(
                ReadinessStatus::from_native_error(&error).code(),
                code,
                "untagged readiness fallback"
            );
        }
    }
}

struct Fixture {
    root: PathBuf,
    state: State,
    comparison: Comparison,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-readiness-kinds-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        let header = json!({"w":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}}).to_string();
        for label in ["a", "b"] {
            let path = root.join(label);
            std::fs::create_dir_all(&path).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0x80, 0x3f, 0, 0xc0]);
            std::fs::write(path.join("tiny.safetensors"), bytes).unwrap();
        }
        let state = State::open(&root.join("a"), &root.join("viewer-cache"), None, None).unwrap();
        let comparison = Comparison::open(
            &root.join("a"),
            &root.join("b"),
            &root.join("comparison-cache"),
        )
        .unwrap();
        Self {
            root,
            state,
            comparison,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn actual_producers_retain_readiness_reason_and_validation_priority() {
    let _isolation = crate::resources::test_workspace_guard();
    hold(
        render::legend("global_linear", None, None).unwrap_err(),
        Readiness::Checkpoint,
        "Full checkpoint calibration is not ready",
    );
    hold(
        render::legend("tensor_linear", None, None).unwrap_err(),
        Readiness::Tensor,
        "Complete selected-tensor calibration is not ready",
    );
    let invalid = render::legend("unknown", None, None).unwrap_err();
    assert_eq!(
        invalid.readiness(),
        None,
        "rule validation precedes readiness"
    );
    assert_eq!(invalid.to_string(), "Unknown color rule");
    let fixture = Fixture::new();
    let tensor = fixture.state.source.tensor(0).unwrap();
    let legend = json!({"max":1.,"histogram_sha256":"different-histogram"});
    hold(
        fixture
            .state
            .tensor_mapping(tensor, "tensor_signed_percentile", &legend)
            .err()
            .unwrap(),
        Readiness::Tensor,
        "Complete selected-tensor calibration is not ready",
    );
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
    hold(
        fixture
            .state
            .tensor_mapping(tensor, "tensor_signed_percentile", &legend)
            .err()
            .unwrap(),
        Readiness::Histogram,
        "Exact percentile histogram is not ready",
    );
    hold(
        fixture.comparison.legend(0, "delta", "linear").unwrap_err(),
        Readiness::Pair,
        "Complete paired-tensor calibration is not ready",
    );
    let invalid = fixture
        .comparison
        .legend(0, "unknown", "linear")
        .unwrap_err();
    assert_eq!(
        invalid.readiness(),
        None,
        "quantity validation precedes readiness"
    );
    assert_eq!(invalid.to_string(), "Unsupported comparison quantity");
}
