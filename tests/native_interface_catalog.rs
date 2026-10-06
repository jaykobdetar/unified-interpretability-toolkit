//! Audited native names held independently of consumer formatting.
use weight_atlas_rust::{
    api,
    command::{Command, COMPARISON_PREFIX},
};

#[test]
fn viewer_paths_methods_and_public_parameters_keep_the_audited_contract() {
    let observed = api::viewer::ROUTES
        .iter()
        .map(|r| (r.path, r.method, r.parameters, r.public))
        .collect::<Vec<_>>();
    assert_eq!(
        observed,
        vec![
            ("/api/model", "GET", &[][..], true),
            ("/api/progress", "GET", &["tensor"][..], true),
            ("/api/tensor-status", "GET", &["tensor"][..], true),
            (
                "/api/view",
                "GET",
                &["tensor", "slice", "left", "right"][..],
                true
            ),
            (
                "/api/inspect",
                "GET",
                &["tensor", "slice", "row", "col", "left", "right"][..],
                true
            ),
            (
                "/tile",
                "GET",
                &["tensor", "slice", "rule", "level", "x", "y", "binding"][..],
                true
            ),
            ("/api/calibrate", "POST", &["tensor", "all"][..], true),
            ("/api/binding", "GET", &["tensor", "slice"][..], false),
        ]
    );
}

#[test]
fn private_names_keep_distinct_tile_calibration_and_binding_sets() {
    for (id, expected) in [
        ("model", &[][..]),
        ("progress", &["tensor"][..]),
        ("tensor-status", &["tensor"][..]),
        ("view", &["tensor", "slice", "left", "right"][..]),
        (
            "inspect",
            &["tensor", "slice", "row", "col", "left", "right"][..],
        ),
        ("tile", &["tensor", "slice", "rule", "level", "x", "y"][..]),
        ("calibration", &["tensor"][..]),
        ("binding", &["tensor", "slice"][..]),
    ] {
        assert_eq!(
            api::hosted::parameters(id),
            Some(expected),
            "private declared set: {id}"
        );
    }
    for id in ["Model", "calibrate", "unknown", ""] {
        assert_eq!(api::hosted::parameters(id), None);
    }
}

#[test]
fn comparison_paths_keep_identity_on_every_route_and_all_refusal_name() {
    let observed = api::comparison::ROUTES
        .iter()
        .map(|r| (r.path, r.method, r.parameters, r.public))
        .collect::<Vec<_>>();
    assert_eq!(
        observed,
        vec![
            (
                "/api/comparison/model",
                "GET",
                &["comparison_identity"][..],
                true
            ),
            (
                "/api/comparison/view",
                "GET",
                &["comparison_identity", "tensor", "left", "right", "mapping"][..],
                true
            ),
            (
                "/api/comparison/inspect",
                "GET",
                &["comparison_identity", "tensor", "row", "col"][..],
                true
            ),
            (
                "/api/comparison/tile",
                "GET",
                &[
                    "comparison_identity",
                    "tensor",
                    "quantity",
                    "mapping",
                    "level",
                    "x",
                    "y"
                ][..],
                true
            ),
            (
                "/api/comparison/calibrate",
                "POST",
                &["comparison_identity", "tensor", "all"][..],
                true
            ),
        ]
    );
}

#[test]
fn all_fifteen_commands_keep_common_hidden_consumed_and_private_names() {
    for (id, specific) in [
        ("metadata", &[][..]),
        ("verify", &[][..]),
        ("serve", &["port", "verify-sha"][..]),
        ("calibrate", &["tensor"][..]),
        ("overview", &["tensor", "slice", "rules", "max-values"][..]),
        (
            "tile",
            &["tensor", "slice", "rules", "level", "x", "y", "out"][..],
        ),
        (
            "inspect",
            &["tensor", "slice", "row", "col", "left", "right"][..],
        ),
        ("bench", &["tensor", "repeats"][..]),
        ("compare-metadata", &["compare-model", "tensor"][..]),
        ("compare-calibrate", &["compare-model", "tensor"][..]),
        ("compare-serve", &["compare-model", "tensor", "port"][..]),
        (
            "compare-tile",
            &[
                "compare-model",
                "tensor",
                "quantity",
                "mapping",
                "level",
                "x",
                "y",
                "out",
            ][..],
        ),
        (
            "compare-inspect",
            &["compare-model", "tensor", "row", "col"][..],
        ),
        ("hosted-renderer", &["channel-fd"][..]),
        ("profile-worker", &["input-fd", "input-sha"][..]),
    ] {
        let options = Command::lookup(id).unwrap().options();
        assert_eq!(
            options.specific, specific,
            "command-specific declared names: {id}"
        );
        assert_eq!(
            options.common,
            if id == "profile-worker" {
                &[
                    "model",
                    "revision",
                    "tensor",
                    "slice",
                    "seed",
                    "values",
                    "wall-ms",
                    "cpu-ms",
                    "binding",
                    "output-fd",
                ][..]
            } else {
                &["model", "cache", "name", "revision", "resources"][..]
            },
            "command common names: {id}"
        );
        let all = options.common.iter().chain(options.specific);
        let expected = options.common.len() + options.specific.len();
        assert_eq!(
            all.collect::<std::collections::BTreeSet<_>>().len(),
            expected
        );
    }
    assert_eq!(COMPARISON_PREFIX, "compare-");
    assert_eq!(Command::lookup("compare-unknown"), None);
}
