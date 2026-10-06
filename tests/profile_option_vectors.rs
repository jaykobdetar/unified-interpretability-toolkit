use std::collections::BTreeMap;
use weight_atlas_rust::profile_worker;

// Every case also has an invalid duration or finalization margin. If the one
// guarded validation is broken, another early refusal still prevents configure,
// source reads, CPU limits, private descriptors and actual profile work.
fn options() -> BTreeMap<String, String> {
    [
        ("model", "/unused"),
        ("revision", "fixture"),
        ("tensor", "0"),
        ("slice", ""),
        ("seed", "17"),
        ("values", "1"),
        ("wall-ms", "200"),
        ("cpu-ms", "0"),
        ("binding", "fixture"),
        ("output-fd", "0"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect()
}

#[test]
fn worker_unknown_options_precede_duration_and_resource_work() {
    let mut opts = options();
    opts.insert("unexpected".into(), "1".into());
    assert_eq!(
        profile_worker::run(&opts).unwrap_err().to_string(),
        "Unknown or missing worker option",
        "worker unknown option refusal"
    );
}

#[test]
fn worker_restore_pair_precedes_duration_and_resource_work() {
    let mut opts = options();
    opts.insert("input-fd".into(), "3".into());
    assert_eq!(
        profile_worker::run(&opts).unwrap_err().to_string(),
        "Incomplete restore input",
        "worker restore correspondence refusal"
    );
}

#[test]
fn worker_metadata_budget_precedes_duration_and_resource_work() {
    let mut opts = options();
    let used = opts.values().map(String::len).sum::<usize>();
    opts.get_mut("revision")
        .unwrap()
        .push_str(&"r".repeat(16385 - used));
    assert_eq!(opts.values().map(String::len).sum::<usize>(), 16385);
    assert_eq!(
        profile_worker::run(&opts).unwrap_err().to_string(),
        "Worker metadata too large",
        "worker metadata boundary refusal"
    );
}

#[test]
fn worker_zero_cpu_grant_precedes_finalization_and_resource_work() {
    assert_eq!(
        profile_worker::run(&options()).unwrap_err().to_string(),
        "Worker duration outside total grant",
        "worker duration admission refusal"
    );
}
