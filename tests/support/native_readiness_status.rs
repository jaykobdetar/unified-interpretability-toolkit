//! Preserve the contextual comparison HTTP fallback, including incidental text.
use super::code;
use std::{cell::Cell, fmt, io};

#[test]
fn readiness_status_preserves_exact_substring_policy() {
    for (message, expected) in [
        ("", 400),
        ("not ready", 503),
        ("Full checkpoint calibration is not ready", 503),
        ("Complete selected-tensor calibration is not ready", 503),
        ("Exact percentile histogram is not ready", 503),
        ("Complete paired-tensor calibration is not ready", 503),
        ("prefix not ready suffix", 503),
        ("雪 not ready λ", 503),
        ("not ready suffix", 503),
        ("prefix not ready", 503),
        ("ready", 400),
        ("not calibrated", 400),
        ("NOT READY", 400),
        ("Not ready", 400),
        ("not Ready", 400),
        ("not  ready", 400),
        ("not\tready", 400),
        ("not\nready", 400),
        ("Unknown tensor", 400),
        ("Numeric queue full; retry shortly", 400),
    ] {
        assert_eq!(
            display_code(&message),
            expected,
            "readiness string: {message:?}"
        );
        let error: crate::Error = message.into();
        assert_eq!(
            code(&error),
            expected,
            "readiness owned refusal: {message:?}"
        );
        let error = io::Error::other(message);
        assert_eq!(
            display_code(&error),
            expected,
            "readiness standard error: {message:?}"
        );
        let native: crate::Error = error.into();
        assert_eq!(
            code(&native),
            expected,
            "readiness native standard error: {message:?}"
        );
    }
}

struct ChangingDisplay(Cell<usize>);
impl fmt::Display for ChangingDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let calls = self.0.get();
        self.0.set(calls + 1);
        f.write_str(if calls == 0 {
            "custom source is not ready yet"
        } else {
            "custom source changed"
        })
    }
}

#[test]
fn readiness_formats_custom_display_once() {
    let error = ChangingDisplay(Cell::new(0));
    assert_eq!(
        display_code(&error),
        503,
        "readiness custom first display determines status"
    );
    assert_eq!(error.0.get(), 1, "readiness formatter called exactly once");
}

// Keep direct generic Display coverage alongside the native error boundary.
fn display_code(error: &dyn fmt::Display) -> u16 {
    crate::http_status::ReadinessStatus::from_error(error).code()
}

#[derive(Debug)]
struct CountedSource(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl fmt::Display for CountedSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let previous = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        f.write_str(if previous == 0 {
            "custom source is not ready yet"
        } else {
            "custom source changed"
        })
    }
}
impl std::error::Error for CountedSource {}

#[test]
fn readiness_formats_native_custom_source_once() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let error: crate::Error = io::Error::other(CountedSource(calls.clone())).into();
    assert_eq!(
        code(&error),
        503,
        "readiness native custom first display determines status"
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "readiness native formatter called exactly once"
    );
}
