//! Exact metadata spelling with an immutable, once-parsed numeric interpretation.
use super::Dtype;
use crate::Result;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, ops::Deref};

const UNSUPPORTED: &str = "Only BF16, F16 and F32 tensors are supported";

mod sealed {
    pub trait Sealed {}
}

/// Inputs accepted by the existing numeric parser. Metadata uses its cached result.
pub trait FormatInput: sealed::Sealed {
    fn parsed_dtype(&self) -> Result<Dtype>;
}

impl sealed::Sealed for str {}
impl FormatInput for str {
    fn parsed_dtype(&self) -> Result<Dtype> {
        match self {
            "BF16" => Ok(Dtype::Bf16),
            "F16" => Ok(Dtype::F16),
            "F32" => Ok(Dtype::F32),
            _ => Err(UNSUPPORTED.into()),
        }
    }
}

impl sealed::Sealed for String {}
impl FormatInput for String {
    fn parsed_dtype(&self) -> Result<Dtype> {
        self.as_str().parsed_dtype()
    }
}

/// Retains unsupported catalog encodings as strings without interpreting their data.
#[derive(Clone, PartialEq, Eq)]
pub struct Format {
    name: String,
    numeric: Option<Dtype>,
}

impl From<String> for Format {
    fn from(name: String) -> Self {
        let numeric = name.parsed_dtype().ok();
        Self { name, numeric }
    }
}

impl From<&str> for Format {
    fn from(name: &str) -> Self {
        Self::from(name.to_owned())
    }
}

impl sealed::Sealed for Format {}
impl FormatInput for Format {
    fn parsed_dtype(&self) -> Result<Dtype> {
        self.numeric.ok_or_else(|| UNSUPPORTED.into())
    }
}

impl Format {
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl Deref for Format {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<&str> for Format {
    fn eq(&self, other: &&str) -> bool {
        self.name == *other
    }
}

impl fmt::Debug for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.name, f)
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

impl Serialize for Format {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.name.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Format {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_cache_survives_clone_and_string_deserialization() {
        for (name, numeric) in [
            ("BF16", Some(Dtype::Bf16)),
            ("F16", Some(Dtype::F16)),
            ("F32", Some(Dtype::F32)),
            ("NEW_CODEC", None),
            ("bf16", None),
            ("Ｆ16", None),
            ("", None),
        ] {
            let value = Format::from(name);
            assert_eq!(value.numeric, numeric);
            assert_eq!(value.clone().numeric, numeric);
            assert_eq!(Dtype::parse(&value).ok(), numeric);
            let raw = serde_json::to_string(name).unwrap();
            let restored: Format = serde_json::from_str(&raw).unwrap();
            assert_eq!(restored.numeric, numeric);
            assert_eq!(serde_json::to_string(&restored).unwrap(), raw);
            assert_eq!(restored.as_str(), name);
            assert_eq!(format!("{restored:?}"), format!("{name:?}"));
            assert_eq!(restored.to_string(), name);
        }
    }

    #[test]
    fn owned_string_allocation_and_legacy_string_parser_remain_usable() {
        let name = String::from("F32");
        let allocation = name.as_ptr();
        assert_eq!(Dtype::parse(&name).unwrap(), Dtype::F32);
        let value = Format::from(name);
        assert_eq!(value.as_bytes().as_ptr(), allocation);
        assert_eq!(value, "F32");
        assert_eq!(Dtype::parse(&value).unwrap(), Dtype::F32);
    }
}
