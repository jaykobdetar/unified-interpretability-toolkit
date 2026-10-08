//! Route field semantics with no source, listener, model execution or descriptor work.
use std::collections::BTreeMap;
use weight_atlas_rust::{api::argument as arg, command::argument as cli, server, Error};

fn values(items: &[(&str, &str)]) -> BTreeMap<String, String> {
    items
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn route_defaults_keep_viewer_comparison_and_calibration_distinct() {
    let q = server::query(&[]);
    assert_eq!(q.read(&arg::TENSOR).unwrap(), 0);
    assert_eq!(q.read(&arg::ROW).unwrap(), 0);
    assert_eq!(q.read(&arg::COL).unwrap(), 0);
    assert_eq!(q.read(&arg::LEVEL).unwrap(), 0);
    assert_eq!(q.read(&arg::X).unwrap(), 0);
    assert_eq!(q.read(&arg::Y).unwrap(), 0);
    assert_eq!(q.read(&arg::SLICE).unwrap(), Vec::<usize>::new());
    assert_eq!(q.field(&arg::SELECTED_TENSOR), "");
    assert_eq!(q.field(&arg::LEFT), "global_linear");
    assert_eq!(q.field(&arg::RIGHT), "global_asinh");
    assert_eq!(q.field(&arg::RULE), "global_linear");
    assert_eq!(q.field(&arg::BINDING), "");
    assert_eq!(q.field(&arg::ALL), "0");
    assert_eq!(q.field(&arg::comparison::COMPARISON_IDENTITY), "");
    assert_eq!(q.field(&arg::comparison::LEFT), "a");
    assert_eq!(q.field(&arg::comparison::RIGHT), "b");
    assert_eq!(q.field(&arg::comparison::QUANTITY), "delta");
    assert_eq!(q.field(&arg::comparison::MAPPING), "linear");
    let error = q.read(&arg::comparison::CALIBRATION_TENSOR).unwrap_err();
    assert!(matches!(error, Error::Integer(_)));
    assert_eq!(error.to_string(), "cannot parse integer from empty string");
}

#[test]
fn route_fields_keep_explicit_empty_unknown_and_numeric_boundaries() {
    let q = server::query(
        &values(&[
            ("tensor", "17"),
            ("row", "3"),
            ("col", "4"),
            ("x", "5"),
            ("y", "6"),
            ("level", "7"),
            ("slice", "2,3"),
            ("left", ""),
            ("unknown", "ignored"),
            ("", "ignored"),
        ])
        .into_iter()
        .collect::<Vec<_>>(),
    );
    assert_eq!(q.read(&arg::TENSOR).unwrap(), 17);
    assert_eq!(q.read(&arg::ROW).unwrap(), 3);
    assert_eq!(q.read(&arg::COL).unwrap(), 4);
    assert_eq!(q.read(&arg::X).unwrap(), 5);
    assert_eq!(q.read(&arg::Y).unwrap(), 6);
    assert_eq!(q.read(&arg::LEVEL).unwrap(), 7);
    assert_eq!(q.read(&arg::SLICE).unwrap(), vec![2, 3]);
    assert_eq!(q.field(&arg::LEFT), "");
    assert_eq!(q.field(&arg::comparison::LEFT), "");
    for (input, message) in [
        ("", "cannot parse integer from empty string"),
        ("-1", "invalid digit found in string"),
        ("no", "invalid digit found in string"),
    ] {
        let q = server::query(&[("tensor".into(), input.into())]);
        let error = q.read(&arg::TENSOR).unwrap_err();
        assert!(matches!(error, Error::Integer(_)));
        assert_eq!(error.to_string(), message);
    }
    if usize::BITS > 32 {
        let q = server::query(&[("level".into(), "4294967296".into())]);
        let error = q.read(&arg::LEVEL).unwrap_err();
        assert!(matches!(error, Error::Range(_)));
        assert_eq!(
            error.to_string(),
            "out of range integral type conversion attempted"
        );
    }
    let q = server::query(&[("slice".into(), "2,,3".into())]);
    assert_eq!(
        q.read(&arg::SLICE).unwrap_err().to_string(),
        "Slice indices must be unsigned decimal integers"
    );
    let q = server::query(&[("slice".into(), "0".repeat(257))]);
    assert_eq!(
        q.read(&arg::SLICE).unwrap_err().to_string(),
        "Slice index list too long"
    );
}

#[test]
fn private_worker_fields_retain_types_without_early_domain_checks() {
    let opts = values(&[
        ("tensor", "8"),
        ("slice", "2,3"),
        ("seed", "9"),
        ("values", "10"),
        ("wall-ms", "700"),
        ("cpu-ms", "600"),
        ("output-fd", "11"),
        ("input-fd", "12"),
    ]);
    assert_eq!(cli::PROFILE_TENSOR.read(&opts).unwrap(), 8);
    assert_eq!(cli::PROFILE_SLICE.read(&opts).unwrap(), vec![2, 3]);
    assert_eq!(cli::PROFILE_SEED.read(&opts).unwrap(), 9);
    assert_eq!(cli::PROFILE_VALUES.read(&opts).unwrap(), 10);
    assert_eq!(cli::PROFILE_WALL_MS.read(&opts).unwrap(), 700);
    assert_eq!(cli::PROFILE_CPU_MS.read(&opts).unwrap(), 600);
    assert_eq!(cli::PROFILE_OUTPUT_FD.read(&opts).unwrap(), 11);
    assert_eq!(cli::PROFILE_INPUT_FD.optional(&opts).unwrap(), Some(12));
    assert_eq!(cli::PROFILE_INPUT_FD.optional(&values(&[])).unwrap(), None);
    assert_eq!(cli::PROFILE_OUTPUT_FD.parse_value("-3").unwrap(), -3);
    assert_eq!(
        cli::PROFILE_SEED.parse_value("4294967296").unwrap(),
        4294967296
    );
    // The unchanged owner validates descriptor ownership and seed bounds later.
    assert!(matches!(
        cli::PROFILE_SEED.parse_value("-1"),
        Err(Error::Integer(_))
    ));
}
