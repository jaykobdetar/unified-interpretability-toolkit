//! Exact CLI grammar and early startup decisions, with no resource/model work.
use super::*;
use weight_atlas_rust::resources::StartupScope;

#[test]
fn command_definition_options_keep_values_order_and_original_refusals() {
    for (args, expected) in [
        (vec!["metadata"], vec![]),
        (
            vec!["tile", "--tensor", "4", "--rules", "tensor_linear"],
            vec![("rules", "tensor_linear"), ("tensor", "4")],
        ),
        (
            vec!["metadata", "--b", "café 雪", "--a", "--literal-value"],
            vec![("a", "--literal-value"), ("b", "café 雪")],
        ),
        (vec!["inspect", "--slice", ""], vec![("slice", "")]),
    ] {
        let args = args.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let outcome = parse_options(&args)
            .map(|options| options.into_iter().collect::<Vec<_>>())
            .map_err(|error| error.to_string());
        assert_eq!(
            outcome,
            Ok(expected
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect::<Vec<_>>()),
            "command definition option values/order"
        );
    }
    for (args, message) in [
        (vec!["metadata", "--cache"], "Options require values"),
        (
            vec!["metadata", "model", "ordinary-value"],
            "Expected --option value",
        ),
        (
            vec!["metadata", "--name", "one", "--name", "two"],
            "Duplicate CLI option",
        ),
    ] {
        let args = args.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let error = parse_options(&args).err();
        assert_eq!(
            error.as_ref().map(ToString::to_string),
            Some(message.to_owned()),
            "command definition grammar refusal"
        );
        assert!(matches!(error, Some(weight_atlas_rust::Error::Refusal(_))));
    }
}

#[test]
fn command_definition_startup_scope_keeps_all_public_private_and_unknown_ids() {
    for command in [
        "metadata",
        "serve",
        "calibrate",
        "verify",
        "tile",
        "overview",
        "inspect",
        "bench",
    ] {
        assert_eq!(
            StartupScope::for_command(command),
            StartupScope::Standalone,
            "command definition standalone: {command}"
        );
    }
    for command in [
        "compare-metadata",
        "compare-calibrate",
        "compare-tile",
        "compare-inspect",
        "compare-serve",
        "hosted-renderer",
        "profile-worker",
        "",
        "--help",
        "Metadata",
        "metadata-extra",
        "unknown",
        "compare-unknown",
    ] {
        assert_eq!(
            StartupScope::for_command(command),
            StartupScope::Legacy,
            "command definition legacy: {command}"
        );
    }
}

#[test]
fn command_definition_intake_keeps_refusal_order_and_configure_boundary() {
    for (args, message) in [
        (vec!["metadata"], "--model DIRECTORY is required"),
        (vec!["unknown"], "--model DIRECTORY is required"),
        (vec!["compare-unknown"], "--model DIRECTORY is required"),
        (vec!["metadata", "--model"], "Options require values"),
    ] {
        let called = std::cell::Cell::new(false);
        let error = run_args(args.into_iter().map(str::to_owned).collect(), |_| {
            called.set(true);
            Err("Unexpected configure boundary".into())
        })
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            message,
            "command definition intake refusal"
        );
        assert!(!called.get(), "command definition early refusal");
    }
    for (command, expected) in [
        ("metadata", StartupScope::Standalone),
        ("unknown", StartupScope::Legacy),
        ("compare-unknown", StartupScope::Legacy),
    ] {
        let observed = std::cell::Cell::new(None);
        let error = run_args(
            vec![command.into(), "--model".into(), "/unused".into()],
            |scope| {
                observed.set(Some(scope));
                Err("Held configure boundary".into())
            },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Held configure boundary",
            "command definition configured refusal"
        );
        assert_eq!(
            observed.get(),
            Some(expected),
            "command definition observed scope"
        );
    }
}
