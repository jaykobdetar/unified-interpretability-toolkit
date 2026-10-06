//! The ordered native rule catalog. Rendering consumes these facts rather than IDs.
use crate::{require, source::Dtype, Result};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scope {
    Checkpoint,
    Tensor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transform {
    SignedLinear,
    SignedAsinh,
    MagnitudeLinear,
    MagnitudeAsinh,
    SignedPercentile,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Statistics {
    Maximum,
    MaximumAndMedian,
    Quantile99,
    Quantile99AndMedian,
    ExactHistogram,
}

pub(crate) struct Definition {
    pub id: &'static str,
    title: &'static str,
    formula: &'static str,
    pub scope: Scope,
    pub transform: Transform,
    pub statistics: Statistics,
    supported_dtypes: &'static [Dtype],
    availability_note: &'static str,
    dtype_refusal: &'static str,
}

/// An owned selection retains the ID's string view at existing caller boundaries.
pub(crate) struct Selected(pub &'static Definition);

impl std::ops::Deref for Selected {
    type Target = str;

    fn deref(&self) -> &str {
        self.0.id
    }
}

const NUMERIC: &[Dtype] = &[Dtype::Bf16, Dtype::F16, Dtype::F32];
const EXACT_16_BIT: &[Dtype] = &[Dtype::Bf16, Dtype::F16];

static DEFINITIONS: [Definition; 8] = [
    Definition {
        id: "global_linear",
        title: "Global linear",
        formula: "clip(x/G, -1, 1)",
        scope: Scope::Checkpoint,
        transform: Transform::SignedLinear,
        statistics: Statistics::Maximum,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "global_asinh",
        title: "Global asinh",
        formula: "clip(asinh(x/s)/asinh(G/s), -1, 1); s = G/100 (all-zero fallback 1)",
        scope: Scope::Checkpoint,
        transform: Transform::SignedAsinh,
        statistics: Statistics::Maximum,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "tensor_linear",
        title: "Tensor linear",
        formula: "clip(x/M, -1, 1)",
        scope: Scope::Tensor,
        transform: Transform::SignedLinear,
        statistics: Statistics::Maximum,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "tensor_asinh",
        title: "Tensor asinh",
        formula: "clip(asinh(x/s)/asinh(M/s), -1, 1); s = median nonzero |x| (all-zero fallback 1)",
        scope: Scope::Tensor,
        transform: Transform::SignedAsinh,
        statistics: Statistics::MaximumAndMedian,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "tensor_magnitude",
        title: "Tensor magnitude",
        formula: "f(x) = abs(x)/M; M = max(abs(x)) over the complete tensor; M = 0 => f(x) = 0",
        scope: Scope::Tensor,
        transform: Transform::MagnitudeLinear,
        statistics: Statistics::Maximum,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "tensor_magnitude_asinh",
        title: "Tensor typical magnitude",
        formula: "min(1, asinh(abs(x)/s)/asinh(D/s)); s = median nonzero |x| (all-zero fallback 1); D = Q99 including zeros (zero-Q99 fallback 1)",
        scope: Scope::Tensor,
        transform: Transform::MagnitudeAsinh,
        statistics: Statistics::Quantile99AndMedian,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "tensor_robust99",
        title: "Tensor robust 99%",
        formula: "clip(x/D, -1, 1); Q99 = linear-interpolated 99th percentile of all original |x|; D = Q99 if Q99 != 0, else 1",
        scope: Scope::Tensor,
        transform: Transform::SignedLinear,
        statistics: Statistics::Quantile99,
        supported_dtypes: NUMERIC,
        availability_note: "",
        dtype_refusal: "",
    },
    Definition {
        id: "tensor_signed_percentile",
        title: "Tensor signed percentile",
        formula: "sign(x) * (count(|X| < |x|) + count(|X| <= |x|))/(2*N); X = complete original tensor including zeros; x = 0 => 0",
        scope: Scope::Tensor,
        transform: Transform::SignedPercentile,
        statistics: Statistics::ExactHistogram,
        supported_dtypes: EXACT_16_BIT,
        availability_note: "F32 exact absolute-value rank index is not implemented; no approximate ranks are substituted.",
        dtype_refusal: "Tensor signed percentile is unsupported for F32: exact absolute-value rank index is pending; no approximation is used",
    },
];

/// Public order is part of metadata and saved viewer state.
pub const IDS: [&str; 8] = [
    DEFINITIONS[0].id,
    DEFINITIONS[1].id,
    DEFINITIONS[2].id,
    DEFINITIONS[3].id,
    DEFINITIONS[4].id,
    DEFINITIONS[5].id,
    DEFINITIONS[6].id,
    DEFINITIONS[7].id,
];
pub(crate) const SIGNED_PERCENTILE: &Definition = &DEFINITIONS[7];

#[derive(Serialize)]
struct Metadata {
    id: &'static str,
    title: &'static str,
    formula: &'static str,
    supported_dtypes: Vec<&'static str>,
    availability_note: &'static str,
}

impl Definition {
    pub fn metadata(&self) -> Result<Value> {
        Ok(serde_json::to_value(Metadata {
            id: self.id,
            title: self.title,
            formula: self.formula,
            supported_dtypes: self
                .supported_dtypes
                .iter()
                .map(|dtype| dtype.name())
                .collect(),
            availability_note: self.availability_note,
        })?)
    }

    pub fn validate_dtype(&self, dtype: Dtype) -> Result<()> {
        require(self.supported_dtypes.contains(&dtype), self.dtype_refusal)
    }

    pub fn unsigned(&self) -> bool {
        matches!(
            self.transform,
            Transform::MagnitudeLinear | Transform::MagnitudeAsinh
        )
    }

    pub fn asinh(&self) -> bool {
        matches!(
            self.transform,
            Transform::SignedAsinh | Transform::MagnitudeAsinh
        )
    }

    pub fn quantile_bound(&self) -> bool {
        matches!(
            self.statistics,
            Statistics::Quantile99 | Statistics::Quantile99AndMedian
        )
    }

    pub fn requires_histogram(&self) -> bool {
        self.statistics == Statistics::ExactHistogram
    }
}

pub(crate) fn find(id: &str) -> Option<&'static Definition> {
    DEFINITIONS.iter().find(|definition| definition.id == id)
}

pub(crate) fn definition(id: &str) -> Result<&'static Definition> {
    find(id).ok_or_else(|| "Unknown color rule".into())
}
