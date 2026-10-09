//! Exact lazy CLI field semantics, independent of models, listeners and resource setup.
use std::collections::BTreeMap;
use weight_atlas_rust::{command::argument as arg, Error};

fn options(items: &[(&str, &str)]) -> BTreeMap<String, String> {
    items
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn declared_command_defaults_and_contexts() {
    let empty = options(&[]);
    assert_eq!(arg::SERVE_PORT.read(&empty).unwrap(), 8775);
    assert_eq!(arg::COMPARE_PORT.read(&empty).unwrap(), 8776);
    assert_eq!(arg::CACHE.read(&empty).unwrap(), "cache");
    assert_eq!(arg::COMPARE_CACHE.read(&empty).unwrap(), "cache-comparison");
    assert_eq!(
        arg::OVERVIEW_RULES.read(&empty).unwrap(),
        "tensor_linear,tensor_asinh"
    );
    assert_eq!(
        arg::TILE_RULES.read(&empty).unwrap(),
        "global_linear,global_asinh"
    );
    assert_eq!(arg::OVERVIEW_MAX_VALUES.read(&empty).unwrap(), 16777216);
    assert_eq!(arg::BENCH_REPEATS.read(&empty).unwrap(), 3);
    assert_eq!(arg::COMPARE_MAPPING.read(&empty).unwrap(), "linear");
    assert_eq!(arg::COMPARE_QUANTITY.read(&empty).unwrap(), "delta");
    assert_eq!(arg::TILE_OUT.read(&empty).unwrap(), "tile");
    assert_eq!(arg::VERIFY_SHA.read(&empty).unwrap(), "false");
    assert_eq!(arg::TILE_X.read(&empty).unwrap(), 0);
    assert_eq!(arg::TILE_Y.read(&empty).unwrap(), 0);
    assert_eq!(arg::COMPARE_ROW.read(&empty).unwrap(), 0);
    assert_eq!(arg::COMPARE_COL.read(&empty).unwrap(), 0);
    assert_eq!(arg::COMPARE_X.read(&empty).unwrap(), 0);
    assert_eq!(arg::COMPARE_Y.read(&empty).unwrap(), 0);
    assert_eq!(arg::TILE_TENSOR.read(&empty).unwrap(), 0);
    assert_eq!(arg::BENCH_TENSOR.read(&empty).unwrap(), 0);
    assert_eq!(arg::COMPARE_TENSOR.read(&empty).unwrap(), 0);
    assert_eq!(
        arg::OVERVIEW_SLICE.read(&empty).unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(arg::TILE_SLICE.read(&empty).unwrap(), Vec::<usize>::new());
}

#[test]
fn declared_command_explicit_values_empty_and_unknown_names() {
    let values = options(&[
        ("tensor", "17"),
        ("port", "1234"),
        ("unknown", "ignored"),
        ("rules", ""),
        ("slice", "2,3"),
        ("level", "4"),
    ]);
    assert_eq!(arg::TILE_TENSOR.read(&values).unwrap(), 17);
    assert_eq!(arg::SERVE_PORT.read(&values).unwrap(), 1234);
    assert_eq!(arg::TILE_RULES.read(&values).unwrap(), "");
    assert_eq!(arg::TILE_SLICE.read(&values).unwrap(), vec![2, 3]);
    assert_eq!(arg::TILE_LEVEL.read_with_default(&values, "7").unwrap(), 4);
    assert_eq!(
        arg::COMPARE_LEVEL
            .read_with_default(&options(&[]), "7")
            .unwrap(),
        7
    );
    assert!(arg::TILE_LEVEL
        .read_with_default(&options(&[]), "bad")
        .is_err());
    let empty = arg::SERVE_PORT.read(&options(&[("port", "")])).unwrap_err();
    assert!(matches!(empty, Error::Integer(_)));
    assert_eq!(empty.to_string(), "cannot parse integer from empty string");
    assert_eq!(arg::NAME.optional(&values).unwrap(), None);
    assert_eq!(
        arg::REVISION
            .optional(&options(&[("revision", "")]))
            .unwrap(),
        Some(String::new())
    );
    assert_eq!(arg::CALIBRATE_TENSOR.optional(&values).unwrap(), Some(17));
    assert_eq!(arg::CALIBRATE_TENSOR.optional(&options(&[])).unwrap(), None);
}

#[test]
fn declared_command_widths_and_required_errors() {
    for (key, input, expected) in [
        ("port", "65536", "number too large to fit in target type"),
        ("port", "-1", "invalid digit found in string"),
    ] {
        let error = arg::SERVE_PORT.read(&options(&[(key, input)])).unwrap_err();
        assert!(matches!(error, Error::Integer(_)));
        assert_eq!(error.to_string(), expected);
    }
    assert!(matches!(
        arg::TILE_LEVEL.read_with_default(&options(&[("level", "4294967296")]), "0"),
        Err(Error::Integer(_))
    ));
    assert_eq!(
        arg::CHANNEL_FD
            .read(&options(&[("channel-fd", "-3")]))
            .unwrap(),
        -3
    );
    let missing = options(&[]);
    for (actual, message) in [
        (
            arg::MODEL.read(&missing).unwrap_err(),
            "--model DIRECTORY is required",
        ),
        (
            arg::COMPARE_MODEL.read(&missing).unwrap_err(),
            "--compare-model DIRECTORY is required for comparison",
        ),
        (
            arg::COMPARE_OUT.read(&missing).unwrap_err(),
            "--out PREFIX required",
        ),
        (
            arg::OVERVIEW_TENSOR.read(&missing).unwrap_err(),
            "Overview requires explicit --tensor ID",
        ),
        (
            arg::CHANNEL_FD.read(&missing).unwrap_err(),
            "Private channel required",
        ),
    ] {
        assert!(matches!(actual, Error::Refusal(_)));
        assert_eq!(actual.to_string(), message);
    }
    assert_eq!(arg::MODEL.read(&options(&[("model", "")])).unwrap(), "");
    assert!(matches!(
        arg::OVERVIEW_TENSOR.read(&options(&[("tensor", "")])),
        Err(Error::Integer(_))
    ));
}
