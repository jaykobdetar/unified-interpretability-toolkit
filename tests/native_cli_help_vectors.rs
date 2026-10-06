//! Exact help bytes from the immutable qualified native checkpoint.
use std::process::Command;

#[test]
fn command_definition_help_keeps_empty_explicit_and_trailing_argument_behavior() {
    for args in [
        vec![],
        vec!["--help"],
        vec!["--help", "ordinary-trailing-token"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(0),
            "command definition help exit"
        );
        assert_eq!(output.stderr, b"", "command definition help stderr");
        assert_eq!(
            output.stdout,
            include_bytes!("fixtures/native-cli-help.txt"),
            "command definition exact help bytes"
        );
    }
}
