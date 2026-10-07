//! Contextual status selection for readiness-sensitive HTTP replies.
//! Other routes retain their own policies; this is not a global error mapping.
use std::fmt::Display;

pub(crate) enum ReadinessStatus {
    BadRequest,
    ServiceUnavailable,
}

impl ReadinessStatus {
    pub(crate) fn from_native_error(error: &crate::Error) -> Self {
        if error.readiness().is_some() {
            Self::ServiceUnavailable
        } else {
            Self::from_error(error)
        }
    }

    pub(crate) fn from_error(error: &dyn Display) -> Self {
        // Preserve incidental/custom error text and exactly one formatting call.
        if error.to_string().contains("not ready") {
            Self::ServiceUnavailable
        } else {
            Self::BadRequest
        }
    }

    pub(crate) fn code(self) -> u16 {
        match self {
            Self::BadRequest => 400,
            Self::ServiceUnavailable => 503,
        }
    }
}
