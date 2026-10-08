//! Lazy typed parameter declarations. Callers retain admission and evaluation order.
use crate::{require, Result};
use std::collections::BTreeMap;

pub struct Parameter<T> {
    pub name: &'static str,
    pub default: &'static str,
    pub missing: Option<&'static str>,
    parse: fn(&str) -> Result<T>,
}

impl<T> Parameter<T> {
    pub const fn new(
        name: &'static str,
        default: &'static str,
        missing: Option<&'static str>,
        parse: fn(&str) -> Result<T>,
    ) -> Self {
        Self {
            name,
            default,
            missing,
            parse,
        }
    }

    /// Borrow a textual value without validating other fields or allocating.
    pub fn value<'a>(&self, values: &'a BTreeMap<String, String>) -> &'a str {
        values
            .get(self.name)
            .map(String::as_str)
            .unwrap_or(self.default)
    }

    /// Parse at the caller's original validation point, including empty values.
    pub fn read(&self, values: &BTreeMap<String, String>) -> Result<T> {
        self.read_with_default(values, self.default)
    }

    /// The caller evaluates source-dependent defaults before entering this method.
    pub fn read_with_default(&self, values: &BTreeMap<String, String>, default: &str) -> Result<T> {
        let value = match values.get(self.name) {
            Some(value) => value.as_str(),
            None => match self.missing {
                Some(message) => return Err(message.into()),
                None => default,
            },
        };
        (self.parse)(value)
    }

    pub fn optional(&self, values: &BTreeMap<String, String>) -> Result<Option<T>> {
        values
            .get(self.name)
            .map(|value| (self.parse)(value))
            .transpose()
    }

    pub fn parse_value(&self, value: &str) -> Result<T> {
        (self.parse)(value)
    }
}

pub fn number<T: std::str::FromStr>(value: &str) -> Result<T>
where
    crate::Error: From<T::Err>,
{
    Ok(value.parse()?)
}

pub fn text(value: &str) -> Result<String> {
    Ok(value.to_owned())
}

pub fn parse_indices(text: &str) -> Result<Vec<usize>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    require(text.len() <= 256, "Slice index list too long")?;
    text.split(',')
        .map(|part| {
            require(
                !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit()),
                "Slice indices must be unsigned decimal integers",
            )?;
            Ok(part.parse()?)
        })
        .collect()
}
