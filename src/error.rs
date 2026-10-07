//! Native error families with original text, debug output and source behavior.
use std::{error::Error as StdError, fmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Readiness {
    Checkpoint,
    Tensor,
    Histogram,
    Pair,
}

impl Readiness {
    fn message(self) -> &'static str {
        match self {
            Self::Checkpoint => "Full checkpoint calibration is not ready",
            Self::Tensor => "Complete selected-tensor calibration is not ready",
            Self::Histogram => "Exact percentile histogram is not ready",
            Self::Pair => "Complete paired-tensor calibration is not ready",
        }
    }

    pub(crate) fn error(self) -> Error {
        Refusal {
            message: self.message().to_owned(),
            readiness: Some(self),
        }
        .into()
    }
}

/// Owned application refusal, including the existing validation/resource messages.
pub struct Refusal {
    message: String,
    readiness: Option<Readiness>,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.message, f)
    }
}

impl fmt::Debug for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.message, f)
    }
}

impl StdError for Refusal {}

// Keep each payload boxed as before. One table defines the families and their
// conversions/forwarding; no wrapper is added to the original source chain.
macro_rules! native_errors {
    ($($variant:ident($payload:ty)),+ $(,)?) => {
        /// Canonical native failures. HTTP status and CLI exit mapping stay at their callers.
        pub enum Error {
            $($variant(Box<$payload>)),+
        }

        impl fmt::Display for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(Self::$variant(cause) => fmt::Display::fmt(cause.as_ref(), f)),+
                }
            }
        }

        impl fmt::Debug for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(Self::$variant(cause) => fmt::Debug::fmt(cause.as_ref(), f)),+
                }
            }
        }

        impl StdError for Error {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                match self {
                    $(Self::$variant(cause) => StdError::source(cause.as_ref())),+
                }
            }
        }

        $(impl From<$payload> for Error {
            fn from(cause: $payload) -> Self {
                Self::$variant(Box::new(cause))
            }
        })+
    };
}

native_errors! {
    Refusal(Refusal),
    Io(std::io::Error),
    Json(serde_json::Error),
    Integer(std::num::ParseIntError),
    Range(std::num::TryFromIntError),
    Array(std::array::TryFromSliceError),
    Allocation(std::collections::TryReserveError),
    Utf8(std::str::Utf8Error),
    OwnedUtf8(std::string::FromUtf8Error),
    Nul(std::ffi::NulError),
    Clock(std::time::SystemTimeError),
}

impl Error {
    pub(crate) fn readiness(&self) -> Option<Readiness> {
        match self {
            Self::Refusal(cause) => cause.readiness,
            _ => None,
        }
    }

    /// Retain source lookup without requiring a trait import at existing callers.
    pub fn source(&self) -> Option<&(dyn StdError + 'static)> {
        StdError::source(self)
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Refusal {
            message,
            readiness: None,
        }
        .into()
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::from(message.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_overflow_preserves_the_original_standard_error() {
        // A u16 count of usize::MAX overflows the capacity preflight; no vector
        // payload is allocated and no actual memory-pressure fault is attempted.
        let outcome: crate::Result<()> = (|| {
            let mut values = Vec::<u16>::new();
            values.try_reserve_exact(usize::MAX)?;
            Ok(())
        })();
        let error = outcome.unwrap_err();
        assert!(matches!(&error, Error::Allocation(_)));
        assert_eq!(
            error.to_string(),
            "memory allocation failed because the computed capacity exceeded the collection's maximum"
        );
        assert_eq!(
            format!("{error:?}"),
            "TryReserveError { kind: CapacityOverflow }"
        );
        assert!(error.source().is_none());
    }
}

#[cfg(test)]
#[path = "../tests/support/native_readiness_kinds.rs"]
mod readiness_kind_vectors;
