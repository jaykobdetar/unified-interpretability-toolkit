//! Original native error conversion bytes and source chains, independent of I/O.
use std::{error::Error as StdError, fmt};
use weight_atlas_rust::Result;

#[derive(Debug)]
struct Inner;
impl fmt::Display for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("inner cause")
    }
}
impl StdError for Inner {}

#[derive(Debug)]
struct Outer;
impl fmt::Display for Outer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("outer cause")
    }
}
impl StdError for Outer {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&Inner)
    }
}

fn hold(
    operation: impl FnOnce() -> Result<()>,
    display: &str,
    debug: &str,
    sources: &[(&str, &str)],
) {
    let error = operation().unwrap_err();
    assert_eq!(error.to_string(), display, "native conversion display");
    assert_eq!(format!("{error:?}"), debug, "native conversion debug");
    let mut got = Vec::new();
    let mut source = error.source();
    while let Some(cause) = source {
        got.push((cause.to_string(), format!("{cause:?}")));
        source = cause.source();
    }
    assert_eq!(
        got,
        sources
            .iter()
            .map(|(display, debug)| (display.to_string(), debug.to_string()))
            .collect::<Vec<_>>(),
        "native conversion source chain"
    );
}

#[test]
fn messages_and_io_keep_display_debug_and_original_source_depth() {
    hold(
        || Err("borrowed café\n\\tail".into()),
        "borrowed café\n\\tail",
        "\"borrowed café\\n\\\\tail\"",
        &[],
    );
    hold(
        || Err(String::from("owned café\n\\tail").into()),
        "owned café\n\\tail",
        "\"owned café\\n\\\\tail\"",
        &[],
    );
    hold(
        || Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "dense input").into()),
        "dense input",
        "Custom { kind: InvalidData, error: \"dense input\" }",
        &[],
    );
    hold(
        || Err(std::io::Error::from_raw_os_error(2).into()),
        "No such file or directory (os error 2)",
        "Os { code: 2, kind: NotFound, message: \"No such file or directory\" }",
        &[],
    );
    hold(
        || Err(std::io::Error::new(std::io::ErrorKind::InvalidData, Outer).into()),
        "outer cause",
        "Custom { kind: InvalidData, error: Outer }",
        &[("inner cause", "Inner")],
    );
}

#[test]
fn json_numeric_array_and_encoding_errors_keep_exact_original_bytes() {
    hold(
        || {
            let _: serde_json::Value = serde_json::from_str("[")?;
            Ok(())
        },
        "EOF while parsing a list at line 1 column 1",
        "Error(\"EOF while parsing a list\", line: 1, column: 1)",
        &[],
    );
    hold(
        || {
            let _: usize = "-1".parse()?;
            Ok(())
        },
        "invalid digit found in string",
        "ParseIntError { kind: InvalidDigit }",
        &[],
    );
    hold(
        || {
            let _ = u8::try_from(256u16)?;
            Ok(())
        },
        "out of range integral type conversion attempted",
        "TryFromIntError(())",
        &[],
    );
    hold(
        || {
            let _ = <[u8; 2]>::try_from(&[1u8][..])?;
            Ok(())
        },
        "could not convert slice to array",
        "TryFromSliceError(())",
        &[],
    );
    hold(
        || {
            let _ = String::from_utf8(vec![255])?;
            Ok(())
        },
        "invalid utf-8 sequence of 1 bytes from index 0",
        "FromUtf8Error { bytes: [255], error: Utf8Error { valid_up_to: 0, error_len: Some(1) } }",
        &[],
    );
    hold(
        || {
            let bytes = std::hint::black_box([255]);
            let _ = std::str::from_utf8(&bytes)?;
            Ok(())
        },
        "invalid utf-8 sequence of 1 bytes from index 0",
        "Utf8Error { valid_up_to: 0, error_len: Some(1) }",
        &[],
    );
    hold(
        || {
            let _ = std::ffi::CString::new("a\0b")?;
            Ok(())
        },
        "nul byte found in provided data at position: 1",
        "NulError(1, [97, 0, 98])",
        &[],
    );
}

#[test]
fn clock_error_keeps_text_and_the_native_error_remains_send_sync() {
    hold(
        || {
            let _ = std::time::UNIX_EPOCH
                .duration_since(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1))?;
            Ok(())
        },
        "second time provided was later than self",
        "SystemTimeError(1s)",
        &[],
    );
    fn send_sync<T: Send + Sync>() {}
    trait ErrorType {
        type Error;
    }
    impl<T, E> ErrorType for std::result::Result<T, E> {
        type Error = E;
    }
    send_sync::<<Result<()> as ErrorType>::Error>();
}
