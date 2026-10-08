//! Compatibility witnesses for later centralization of native rule definitions.
//! The metadata fixture comes from the unchanged original-binary lock baseline.
use serde_json::Value;
use weight_atlas_rust::{
    render::{self, Stats},
    source::Dtype,
};

fn stats(dtype: Dtype) -> Stats {
    Stats {
        count: 8,
        max_abs: 4.,
        median_nonzero_abs: 2.,
        q99: 2.,
        robust_clipped_count: 1,
        histogram_sha256: Some("fixed-histogram".into()),
        exact_zero_count: 2,
        unique_bit_patterns: Some(6),
        dtype: dtype.name().into(),
        calibration_method: "inert-contract-vector".into(),
        seconds: 0.,
    }
}

#[test]
fn registered_rule_metadata_matches_original_binary_vectors() {
    let expected: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/rule_metadata_vectors.json")).unwrap();
    let names = expected
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(render::RULES.as_slice(), names);
    for row in expected {
        let id = row["id"].as_str().unwrap();
        assert_eq!(
            serde_json::to_vec(&render::rule_info(id).unwrap()).unwrap(),
            serde_json::to_vec(&row).unwrap(),
            "{id}"
        );
    }
}

#[test]
fn known_rules_preserve_dtype_admission_and_exact_refusals() {
    let expected: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/rule_metadata_vectors.json")).unwrap();
    for row in expected {
        let rule = row["id"].as_str().unwrap();
        for dtype in [Dtype::Bf16, Dtype::F16, Dtype::F32] {
            let admitted = row["supported_dtypes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|name| name.as_str() == Some(dtype.name()));
            let result = render::validate_rule_dtype(rule, dtype);
            assert_eq!(result.is_ok(), admitted, "{rule} {dtype:?}");
            if !admitted {
                assert_eq!(result.unwrap_err().to_string(), "Tensor signed percentile is unsupported for F32: exact absolute-value rank index is pending; no approximation is used");
            }
        }
    }
    for rule in ["unknown", "Tensor_linear", " tensor_linear"] {
        assert_eq!(
            render::rule_info(rule).unwrap_err().to_string(),
            "Unknown color rule"
        );
        for dtype in [Dtype::Bf16, Dtype::F16, Dtype::F32] {
            let result = render::validate_rule_dtype(rule, dtype);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err().to_string(), "Unknown color rule");
        }
    }
}

#[test]
fn legends_preserve_fixed_scope_bound_scale_and_palette_vectors() {
    let stats = stats(Dtype::Bf16);
    for (rule, minimum, maximum, scale, scope) in [
        ("global_linear", -8., 8., None, "complete checkpoint"),
        ("global_asinh", -8., 8., Some(0.08), "complete checkpoint"),
        ("tensor_linear", -4., 4., None, "complete original tensor"),
        (
            "tensor_asinh",
            -4.,
            4.,
            Some(2.),
            "complete original tensor",
        ),
        ("tensor_magnitude", 0., 4., None, "complete original tensor"),
        (
            "tensor_magnitude_asinh",
            0.,
            2.,
            Some(2.),
            "complete original tensor",
        ),
        ("tensor_robust99", -2., 2., None, "complete original tensor"),
        (
            "tensor_signed_percentile",
            -1.,
            1.,
            None,
            "complete original tensor",
        ),
    ] {
        let legend = render::legend(rule, Some(&stats), Some(8.)).unwrap();
        assert_eq!(legend["min"].as_f64(), Some(minimum), "{rule}");
        assert_eq!(legend["max"].as_f64(), Some(maximum), "{rule}");
        assert_eq!(legend["s"].as_f64(), scale, "{rule}");
        assert_eq!(legend["zero"], 0.);
        assert_eq!(legend["scope"], scope);
    }
    let magnitude = render::legend("tensor_magnitude", Some(&stats), None).unwrap();
    assert_eq!(magnitude["palette"], "sequential-purple-v1");
    assert_eq!(magnitude["field_min"], 0.);
    assert_eq!(magnitude["field_max"], 1.);
}

#[test]
fn legends_preserve_quantile_rank_and_zero_fallback_contracts() {
    let mut stats = stats(Dtype::Bf16);
    for rule in ["tensor_robust99", "tensor_magnitude_asinh"] {
        let legend = render::legend(rule, Some(&stats), None).unwrap();
        assert_eq!(legend["q99"], 2.);
        assert_eq!(legend["effective_divisor"], 2.);
        assert_eq!(legend["zero_quantile_fallback"], false);
        assert_eq!(legend["clipped_count"], 1);
        assert_eq!(legend["clipped_fraction"], 0.125);
        assert_eq!(legend["quantile_order_statistics"], "exact");
    }
    let rank = render::legend("tensor_signed_percentile", Some(&stats), None).unwrap();
    assert_eq!(rank["histogram_sha256"], "fixed-histogram");
    assert_eq!(rank["rank_approximation"], false);
    assert_eq!(
        rank["rank_method"],
        "exact absolute-value mid-CDF including zeros in N"
    );
    stats.max_abs = 0.;
    stats.q99 = 0.;
    stats.median_nonzero_abs = 0.;
    stats.robust_clipped_count = 0;
    for rule in ["tensor_robust99", "tensor_magnitude_asinh"] {
        let legend = render::legend(rule, Some(&stats), None).unwrap();
        assert_eq!(legend["effective_divisor"], 1.);
        assert_eq!(legend["zero_quantile_fallback"], true);
        assert_eq!(legend["clipped_fraction"], 0.);
    }
    for rule in ["global_asinh", "tensor_asinh", "tensor_magnitude_asinh"] {
        let legend = render::legend(rule, Some(&stats), Some(0.)).unwrap();
        assert_eq!(legend["s"], 1.);
    }
    let magnitude = render::legend("tensor_magnitude", Some(&stats), None).unwrap();
    assert_eq!(magnitude["field_max"], 0.);
}

#[test]
fn missing_calibration_and_exact_histogram_refusals_keep_their_text() {
    for rule in ["global_linear", "global_asinh"] {
        assert_eq!(
            render::legend(rule, None, None).unwrap_err().to_string(),
            "Full checkpoint calibration is not ready"
        );
    }
    for rule in ["tensor_linear", "tensor_asinh", "tensor_signed_percentile"] {
        assert_eq!(
            render::legend(rule, None, Some(8.))
                .unwrap_err()
                .to_string(),
            "Complete selected-tensor calibration is not ready"
        );
    }
    let stats = stats(Dtype::Bf16);
    let legend = render::legend("tensor_signed_percentile", Some(&stats), None).unwrap();
    let result = render::mapping("tensor_signed_percentile", &legend, Dtype::Bf16);
    assert!(result.is_err());
    assert_eq!(
        result.err().unwrap().to_string(),
        "Percentile mapping requires an exact histogram"
    );
}

#[test]
fn mappings_preserve_signed_magnitude_clipping_and_asinh_endpoint_vectors() {
    for (dtype, negative, positive, negative_zero, infinity) in [
        (Dtype::Bf16, 0xc080, 0x4080, 0x8000, 0x7f80),
        (Dtype::F16, 0xc400, 0x4400, 0x8000, 0x7c00),
        (
            Dtype::F32,
            0xc080_0000,
            0x4080_0000,
            0x8000_0000,
            0x7f80_0000,
        ),
    ] {
        let stats = stats(dtype);
        for (rule, expected_negative, expected_zero) in [
            ("global_linear", -1., -0.0),
            ("global_asinh", -1., -0.0),
            ("tensor_linear", -1., -0.0),
            ("tensor_asinh", -1., -0.0),
            ("tensor_magnitude", 1., 0.0),
            ("tensor_magnitude_asinh", 1., 0.0),
            ("tensor_robust99", -1., -0.0),
        ] {
            let legend = render::legend(rule, Some(&stats), Some(4.)).unwrap();
            let mapping = render::mapping(rule, &legend, dtype).unwrap();
            assert_eq!(
                mapping.value(negative),
                expected_negative,
                "{rule} {dtype:?}"
            );
            assert_eq!(mapping.value(positive), 1., "{rule} {dtype:?}");
            assert_eq!(
                mapping.value(negative_zero).to_bits(),
                f64::to_bits(expected_zero),
                "{rule} {dtype:?}"
            );
            assert_eq!(mapping.value(0).to_bits(), 0f64.to_bits());
            assert_eq!(mapping.value(infinity).to_bits(), 0f64.to_bits());
        }
        let legend = render::legend("tensor_linear", Some(&stats), None).unwrap();
        let mapping = render::mapping("tensor_linear", &legend, dtype).unwrap();
        let half = if dtype == Dtype::F32 {
            0x4000_0000
        } else {
            0x4000
        };
        assert_eq!(mapping.value(half), 0.5);
    }
}
