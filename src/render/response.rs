//! Private legend presentation; validation and scale selection stay in the caller.
use super::Stats;
use crate::rules::{Definition, Transform};
use serde_json::{json, Value};

pub(super) struct Legend<'a> {
    pub l: Value,
    pub definition: &'a Definition,
    pub stats: Option<&'a Stats>,
    pub g: bool,
    pub bound: f64,
    pub s: Option<f64>,
}

impl From<Legend<'_>> for Value {
    fn from(report: Legend<'_>) -> Self {
        let Legend {
            mut l,
            definition,
            stats,
            g,
            bound,
            s,
        } = report;
        let m = l.as_object_mut().unwrap();
        for(k,v)in json!({"min":-bound,"zero":0.,"max":bound,"s":s,"scope":if g{"complete checkpoint"}else{"complete original tensor"},"units":if s.is_some(){"raw weight on nonlinear scale"}else{"raw weight"},"clipped_fraction":0.,"color_warning":"Finite 8-bit colors merge nearby weights; neutral does not prove exact zero."}).as_object().unwrap(){m.insert(k.clone(),v.clone());}
        if definition.unsigned() {
            m.insert("min".into(), json!(0.));
            m.insert("units".into(), json!("native: absolute raw weight |x|; pooled: mean absolute raw weight mean(|x|); normalized field = mean(|x|)/M"));
            m.insert("palette".into(), json!("sequential-purple-v1"));
            m.insert("field_min".into(), json!(0.));
            m.insert("field_max".into(), json!(if bound == 0. { 0. } else { 1. }));
            m.insert("color_warning".into(), json!("Unsigned magnitude discards sign only for color; original values stay signed. Finite 8-bit colors merge nearby magnitudes; light color does not prove exact zero."));
        }
        if definition.quantile_bound() {
            let stats = stats.unwrap();
            m.insert("q99".into(), json!(stats.q99));
            m.insert("quantile_order_statistics".into(), json!("exact"));
            m.insert(
                "quantile_interpolation".into(),
                json!("linear in F64; final floating-point rounding possible"),
            );
            m.insert("effective_divisor".into(), json!(bound));
            m.insert("zero_quantile_fallback".into(), json!(stats.q99 == 0.));
            m.insert(
                "clipped_fraction".into(),
                json!(stats.robust_clipped_count as f64 / stats.count as f64),
            );
            m.insert("clipped_count".into(), json!(stats.robust_clipped_count));
            m.insert("units".into(), json!("native: raw weight clipped at effective divisor D; pooled: mean of clipped x/D (not a raw weight mean)"));
        }
        if definition.transform == Transform::MagnitudeAsinh {
            m.insert("units".into(), json!("native: transformed absolute weight on median-nonzero asinh scale; pooled: mean of transformed magnitudes, not mean raw magnitude"));
            m.insert("control_semantics".into(), json!("Pointwise permutation-equivariant rule; use identical original-tensor calibration for a matched shuffled field"));
        }
        if definition.requires_histogram() {
            m.insert(
                "rank_method".into(),
                json!("exact absolute-value mid-CDF including zeros in N"),
            );
            m.insert("rank_approximation".into(), json!(false));
            m.insert(
                "histogram_sha256".into(),
                json!(stats.unwrap().histogram_sha256),
            );
            m.insert("units".into(), json!("native: signed absolute-magnitude mid-percentile, dimensionless; pooled: mean signed percentile, not a raw weight or the percentile of a pooled weight"));
            m.insert("color_warning".into(), json!("Tied magnitudes share a percentile; both signed zeros map to zero. Rank colors hide absolute magnitudes and distances."));
        }
        l
    }
}
