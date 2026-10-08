//! Held guard errors and fatal CLI mapping without model or resource work.
use std::process::Command;
use weight_atlas_rust::require;

#[test]
fn require_keeps_admission_and_exact_owned_message() {
    for message in [
        "",
        "Only BF16, F16 and F32 tensors are supported",
        "bad \"rule\"\n雪\\tail",
    ] {
        assert_eq!(
            require(true, message).err().map(|e| e.to_string()),
            None,
            "require true admission"
        );
        let outcome = require(false, message).err().map(|e| e.to_string());
        assert_eq!(outcome, Some(message.to_owned()), "require exact refusal");
    }
    let owned = {
        let message = String::from("borrowed café message");
        require(false, &message).unwrap_err()
    };
    assert_eq!(
        owned.to_string(),
        "borrowed café message",
        "require owned error text"
    );
    assert_eq!(
        format!("{owned:?}"),
        "\"borrowed café message\"",
        "require error debug"
    );
    assert!(owned.source().is_none());
}

#[test]
fn fatal_cli_keeps_exit_stdout_and_exact_stderr_before_model_access() {
    for (args, message) in [
        (vec!["metadata"], "--model DIRECTORY is required"),
        (vec!["metadata", "--cache"], "Options require values"),
        (vec!["metadata", "tensor", "0"], "Expected --option value"),
        (
            vec!["metadata", "--cache", "x", "--cache", "y"],
            "Duplicate CLI option",
        ),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "fatal CLI exit");
        assert_eq!(result.stdout, b"", "fatal CLI stdout");
        assert_eq!(
            String::from_utf8(result.stderr).unwrap(),
            format!("ERROR: {message}\n"),
            "fatal CLI stderr"
        );
    }
    let help = Command::new(env!("CARGO_BIN_EXE_weight-atlas-rust"))
        .arg("--help")
        .output()
        .unwrap();
    assert_eq!(help.status.code(), Some(0), "CLI help success exit");
    assert_eq!(help.stderr, b"");
    assert!(help.stdout.starts_with(b"Weight Atlas Rust\nCommands: "));
}
