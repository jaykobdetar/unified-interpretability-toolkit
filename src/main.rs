#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::Write,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use weight_atlas_rust::{
    atomic_write,
    command::{defaults, parse_options, Command, COMPARISON_PREFIX, HELP},
    configure, configure_standalone, headroom, render, require, server,
    slice::{parse_indices, TensorSlice},
    source::{Dtype, Source},
    state::State,
    Result,
};
#[path = "main/response.rs"]
mod response;

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1)
    }
}
fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    run_args(args, |scope| match scope {
        weight_atlas_rust::resources::StartupScope::Legacy => configure(),
        weight_atlas_rust::resources::StartupScope::Standalone => configure_standalone(),
    })
}
fn run_args(
    args: Vec<String>,
    configure: impl FnOnce(weight_atlas_rust::resources::StartupScope) -> Result<usize>,
) -> Result<()> {
    if args.is_empty() || args[0] == "--help" {
        println!("{HELP}");
        return Ok(());
    }
    let command = &args[0];
    let opts = parse_options(&args)?;
    initialize_resources(command, &opts)?;
    if Command::lookup(command) == Some(Command::ProfileWorker) {
        return weight_atlas_rust::profile_worker::run(&opts);
    }
    let get = |k: &str, default: &str| opts.get(k).cloned().unwrap_or_else(|| default.into());
    let model = PathBuf::from(opts.get("model").ok_or("--model DIRECTORY is required")?);
    let cpu = configure(weight_atlas_rust::resources::StartupScope::for_command(
        command,
    ))?;
    let start = Instant::now();
    if Command::lookup(command) == Some(Command::Metadata) {
        return print_metadata(&model, cpu, start);
    }
    if command.starts_with(COMPARISON_PREFIX) {
        return run_comparison(command, &opts, &model, cpu);
    }
    let state = Arc::new(State::open(
        &model,
        Path::new(&get("cache", defaults::INTAKE_CACHE)),
        opts.get("name").cloned(),
        opts.get("revision").cloned(),
    )?);
    if Command::lookup(command) == Some(Command::Serve)
        && get("verify-sha", defaults::INTAKE_VERIFY_SHA) == "true"
    {
        verify_source(&state, cpu, Instant::now())?;
    }
    dispatch_command(command, &opts, state, cpu, start)
}

fn dispatch_command(
    command: &str,
    opts: &BTreeMap<String, String>,
    state: Arc<State>,
    cpu: usize,
    start: Instant,
) -> Result<()> {
    let get = |k: &str, default: &str| opts.get(k).cloned().unwrap_or_else(|| default.into());
    match Command::lookup(command) {
        Some(Command::Serve) => server::serve(state, get("port", defaults::SERVE_PORT).parse()?)?,
        Some(Command::HostedRenderer) => weight_atlas_rust::hosted_renderer::run(
            state,
            opts.get("channel-fd")
                .ok_or("Private channel required")?
                .parse()?,
        )?,
        Some(Command::Calibrate) => run_calibration(&state, opts, cpu, start)?,
        Some(Command::Verify) => println!("{}", verify_source(&state, cpu, Instant::now())?),
        Some(Command::Inspect) => println!(
            "{}",
            server::inspect(
                &state,
                &server::query(
                    &opts
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<Vec<_>>()
                )
            )?
        ),
        Some(Command::Overview) => run_overview(&state, opts)?,
        Some(Command::Tile) => run_tile(&state, opts, start)?,
        Some(Command::Bench) => run_benchmark(&state, opts, cpu, start)?,
        _ => return Err("Unknown command; use --help".into()),
    };
    Ok(())
}

fn run_calibration(
    state: &State,
    opts: &BTreeMap<String, String>,
    cpu: usize,
    start: Instant,
) -> Result<()> {
    let ids = if let Some(id) = opts.get("tensor") {
        vec![id.parse()?]
    } else {
        state
            .source
            .tensors
            .iter()
            .filter(|t| t.available)
            .map(|t| t.id)
            .collect()
    };
    let mut minimum = headroom()?;
    for id in ids {
        minimum = minimum.min(headroom()?);
        state.calibrate_one(id)?
    }
    let model = state.model()?;
    println!(
        "{}",
        Value::from(response::Calibration {
            model: &model,
            start,
            minimum,
            cpu
        })
    );
    Ok(())
}

fn run_overview(state: &State, opts: &BTreeMap<String, String>) -> Result<()> {
    let get = |k: &str, default: &str| opts.get(k).cloned().unwrap_or_else(|| default.into());
    let id = opts
        .get("tensor")
        .ok_or("Overview requires explicit --tensor ID")?
        .parse()?;
    let leading = parse_indices(&get("slice", defaults::OVERVIEW_SLICE))?;
    let rules = get("rules", defaults::OVERVIEW_RULES);
    let rules = rules.split(',').collect::<Vec<_>>();
    let report = state.prepare_overview(
        id,
        &leading,
        &rules,
        get("max-values", defaults::OVERVIEW_MAX_VALUES).parse()?,
    )?;
    println!("{}", report);
    Ok(())
}

fn run_tile(state: &State, opts: &BTreeMap<String, String>, start: Instant) -> Result<()> {
    let get = |k: &str, default: &str| opts.get(k).cloned().unwrap_or_else(|| default.into());
    let id: usize = get("tensor", defaults::TILE_TENSOR).parse()?;
    let slice = TensorSlice::new(
        &state.source,
        id,
        &parse_indices(&get("slice", defaults::TILE_SLICE))?,
    )?;
    let t = &slice.tensor;
    let level: u32 = get("level", &t.max_level.to_string()).parse()?;
    let rules = get("rules", defaults::TILE_RULES);
    let rules = rules.split(',').collect::<Vec<_>>();
    let luts = rules
        .iter()
        .map(|r| {
            render::validate_rule_dtype(r, Dtype::parse(&t.dtype)?)?;
            render::legend(r, state.stats(id).as_ref(), state.global())
                .and_then(|l| state.tensor_mapping(t, r, &l))
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = luts.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
    let (fields, m) = render::tile_fields(
        &state.source,
        t,
        &refs,
        level,
        get("x", defaults::TILE_X).parse()?,
        get("y", defaults::TILE_Y).parse()?,
    )?;
    let out = get("out", defaults::TILE_OUT);
    for (rule, field) in rules.iter().zip(&fields) {
        std::fs::write(
            format!("{out}-{rule}.png"),
            render::png_for_rule(rule, field, m.width, m.height)?,
        )?;
        let mut f = std::fs::File::create(format!("{out}-{rule}.f64le"))?;
        for x in field {
            f.write_all(&x.to_le_bytes())?
        }
    }
    println!("{}", Value::from(response::Tile { m: &m, start }));
    Ok(())
}

fn run_benchmark(
    state: &State,
    opts: &BTreeMap<String, String>,
    cpu: usize,
    start: Instant,
) -> Result<()> {
    let get = |k: &str, default: &str| opts.get(k).cloned().unwrap_or_else(|| default.into());
    let id = get("tensor", defaults::BENCH_TENSOR).parse()?;
    let t = state.source.tensor(id)?;
    let reps: usize = get("repeats", defaults::BENCH_REPEATS).parse()?;
    require((1..=30).contains(&reps), "repeats 1–30")?;
    let mut records = Vec::new();
    for f in [1usize, 4, 16] {
        if f > 1usize << t.max_level {
            continue;
        }
        let level = t.max_level - f.trailing_zeros();
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..reps {
            let begin = Instant::now();
            let luts = render::RULES[..2]
                .iter()
                .map(|r| {
                    render::legend(r, state.stats(id).as_ref(), state.global())
                        .and_then(|l| state.tensor_mapping(t, r, &l))
                })
                .collect::<Result<Vec<_>>>()?;
            let refs = luts.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
            let (fields, m) = render::tile_fields(&state.source, t, &refs, level, 0, 0)?;
            let mut bytes = 0;
            for field in fields {
                bytes += render::png(&field, m.width, m.height)?.len()
            }
            times.push(begin.elapsed().as_secs_f64());
            last = Some(Value::from(response::BenchmarkSample { m: &m, bytes }));
        }
        records.push(Value::from(response::BenchmarkRun {
            t,
            f,
            level,
            times: &times,
            last: &last,
        }));
    }
    println!(
        "{}",
        Value::from(response::Benchmark {
            records: &records,
            start,
            cpu
        })
    );
    Ok(())
}

fn initialize_resources(command: &str, opts: &BTreeMap<String, String>) -> Result<()> {
    if let Some(raw) = opts.get("resources") {
        require(
            weight_atlas_rust::resources::StartupScope::for_command(command)
                == weight_atlas_rust::resources::StartupScope::Standalone,
            "Custom resources apply only to the standalone viewer/reader",
        )?;
        require(raw.len() <= 4096, "Resource configuration exceeds limit")?;
        weight_atlas_rust::resources::initialize(serde_json::from_str(raw)?)?;
    }
    Ok(())
}

fn print_metadata(model: &Path, cpu: usize, start: Instant) -> Result<()> {
    let s = Source::open(model)?;
    println!("{}", Value::from(response::Metadata { s: &s, start, cpu }));
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn startup_dispatch_keeps_hosted_and_comparison_on_legacy_configuration() {
        use weight_atlas_rust::resources::StartupScope;
        for (command, expected) in [
            ("serve", StartupScope::Standalone),
            ("metadata", StartupScope::Standalone),
            ("hosted-renderer", StartupScope::Legacy),
            ("compare-serve", StartupScope::Legacy),
            ("compare-metadata", StartupScope::Legacy),
        ] {
            let observed = std::cell::Cell::new(None);
            let result = run_args(
                vec![command.into(), "--model".into(), "/unused".into()],
                |scope| {
                    observed.set(Some(scope));
                    Err("Injected startup boundary".into())
                },
            );
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Injected startup boundary"));
            assert_eq!(observed.get(), Some(expected));
        }
        for command in ["hosted-renderer", "compare-serve", "profile-worker"] {
            let called = std::cell::Cell::new(false);
            assert!(run_args(
                vec![
                    command.into(),
                    "--model".into(),
                    "/unused".into(),
                    "--resources".into(),
                    "{}".into()
                ],
                |_| {
                    called.set(true);
                    Err("Unexpected startup".into())
                }
            )
            .unwrap_err()
            .to_string()
            .contains("standalone"));
            assert!(!called.get());
        }
    }
    #[test]
    fn help_and_argument_errors_precede_resource_or_model_work() {
        for args in [vec![], vec!["--help".into()]] {
            let called = std::cell::Cell::new(false);
            run_args(args, |_| {
                called.set(true);
                Err("Injected low-memory guard".into())
            })
            .unwrap();
            assert!(!called.get());
        }
        let called = std::cell::Cell::new(false);
        assert!(run_args(vec!["metadata".into()], |_| {
            called.set(true);
            Err("Injected low-memory guard".into())
        })
        .is_err());
        assert!(!called.get());
        let called = std::cell::Cell::new(false);
        let result = run_args(
            vec!["metadata".into(), "--model".into(), "/unused".into()],
            |_| {
                called.set(true);
                Err("Injected low-memory guard".into())
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Injected low-memory guard"));
        assert!(called.get());
    }
}

fn run_comparison(
    command: &str,
    opts: &BTreeMap<String, String>,
    model: &Path,
    cpu: usize,
) -> Result<()> {
    use weight_atlas_rust::{comparison::Comparison, comparison_http};
    let get = |k: &str, default: &str| opts.get(k).cloned().unwrap_or_else(|| default.into());
    let b = PathBuf::from(
        opts.get("compare-model")
            .ok_or("--compare-model DIRECTORY is required for comparison")?,
    );
    let pair = Arc::new(Comparison::open(
        model,
        &b,
        Path::new(&get("cache", defaults::COMPARISON_CACHE)),
    )?);
    let id = get("tensor", defaults::COMPARISON_TENSOR).parse()?;
    let output = match Command::lookup(command) {
        Some(Command::CompareMetadata) => pair.model()?,
        Some(Command::CompareCalibrate) => {
            require(
                opts.contains_key("tensor"),
                "Comparison calibration requires an explicit --tensor ID",
            )?;
            pair.calibrate_one(id)?;
            pair.model()?
        }
        Some(Command::CompareInspect) => pair.inspect(
            id,
            get("row", defaults::COMPARISON_ROW).parse()?,
            get("col", defaults::COMPARISON_COL).parse()?,
        )?,
        Some(Command::CompareTile) => {
            let quantity = get("quantity", defaults::COMPARISON_QUANTITY);
            let mapping = get("mapping", defaults::COMPARISON_MAPPING);
            let level = get("level", &pair.pair(id)?.max_level.to_string()).parse()?;
            let x = get("x", defaults::COMPARISON_X).parse()?;
            let y = get("y", defaults::COMPARISON_Y).parse()?;
            let prefix =
                pair.output_prefix(Path::new(opts.get("out").ok_or("--out PREFIX required")?))?;
            let (field, metrics) = pair.fields(id, &quantity, &mapping, level, x, y)?;
            let rule = if quantity == "abs_delta" || mapping == "magnitude" {
                "tensor_magnitude"
            } else {
                "tensor_linear"
            };
            let png = render::png_for_rule(rule, &field, metrics.width, metrics.height)?;
            let parent = prefix
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            std::fs::create_dir_all(parent)?;
            let stem = prefix.to_string_lossy();
            atomic_write(Path::new(&format!("{stem}.png")), &png)?;
            atomic_write(
                Path::new(&format!("{stem}.f64le")),
                &field
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            Value::try_from(response::ComparisonTile {
                pair: &pair,
                id,
                quantity: &quantity,
                mapping: &mapping,
                metrics: &metrics,
                cpu,
            })?
        }
        Some(Command::CompareServe) => {
            comparison_http::serve(pair, get("port", defaults::COMPARISON_PORT).parse()?)?;
            return Ok(());
        }
        _ => return Err("Unknown comparison command".into()),
    };
    println!("{}", output);
    Ok(())
}

fn verify_source(state: &State, cpu: usize, start: Instant) -> Result<Value> {
    use sha2::{Digest, Sha256};
    let mut records = Vec::new();
    let mut buf = vec![0u8; 2 * 1048576];
    let mut minimum = headroom()?;
    let ap = state.source.root.join("model-api.json");
    let official: Option<Value> = if ap.exists() {
        Some(weight_atlas_rust::source::json_unique(
            &weight_atlas_rust::source::read_small(&ap, 8 * 1024 * 1024)?,
        )?)
    } else {
        None
    };
    for (s, f) in state.source.shards.iter().zip(&state.source.files) {
        let mut h = Sha256::new();
        let mut offset = 0;
        while offset < s.fingerprint.size {
            minimum = minimum.min(headroom()?);
            let n = (s.fingerprint.size - offset).min(buf.len() as u64) as usize;
            f.read_exact_at(&mut buf[..n], offset)?;
            h.update(&buf[..n]);
            offset += n as u64
        }
        state.source.check()?;
        let hash = format!("{:x}", h.finalize());
        let expected = official
            .as_ref()
            .and_then(|v| v["siblings"].as_array())
            .and_then(|a| a.iter().find(|v| v["rfilename"] == s.name))
            .and_then(|v| v["lfs"]["sha256"].as_str());
        if let Some(e) = expected {
            require(
                hash == e,
                "Full shard SHA mismatch against local pinned metadata",
            )?
        }
        records.push(Value::from(response::VerificationShard {
            s,
            hash: &hash,
            expected,
        }));
        eprintln!("Hashed {}", s.name);
    }
    let v = Value::from(response::Verification {
        source_identity: &state.source.identity,
        records: &records,
        start,
        minimum,
        cpu,
    });
    atomic_write(
        &state.root.join("verification.json"),
        &serde_json::to_vec_pretty(&v)?,
    )?;
    *state.fresh_hashes.lock().unwrap() = Some(v.clone());
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_hash_pass_counts_missing_partial_and_complete_expectations() {
        let root = std::env::temp_dir().join(format!("atlas-hash-scope-{}", std::process::id()));
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let mut siblings = Vec::new();
        for id in 0..3 {
            let header = format!(
                "{{\"tensor{id}\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}}}"
            );
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0x80, 0x3f]);
            let filename = format!("shard-{id}.safetensors");
            std::fs::write(model.join(&filename), &bytes).unwrap();
            siblings.push(
                json!({"rfilename":filename,"lfs":{"sha256":weight_atlas_rust::sha(&bytes)}}),
            );
        }
        for (trial, expected_count) in [None, Some(0), Some(1), Some(3)].iter().enumerate() {
            if let Some(n) = expected_count {
                std::fs::write(
                    model.join("model-api.json"),
                    json!({"siblings":&siblings[..*n]}).to_string(),
                )
                .unwrap();
            }
            let cache = root.join(format!("cache-{trial}"));
            let state = State::open(&model, &cache, None, None).unwrap();
            assert_eq!(state.model().unwrap()["coverage"]["sha_hashed_shards"], 0);
            let report = verify_source(&state, 0, Instant::now()).unwrap();
            assert_eq!(report["shards"].as_array().unwrap().len(), 3);
            let coverage = state.model().unwrap()["coverage"].clone();
            assert_eq!(coverage["sha_hashed_shards"], 3);
            assert_eq!(
                coverage["sha_expected_matched_shards"],
                expected_count.unwrap_or(0)
            );
            assert_eq!(coverage["sha_verified_shards"], expected_count.unwrap_or(0));
            assert_eq!(
                coverage["sha_missing_expected_shards"],
                3 - expected_count.unwrap_or(0)
            );
            drop(state);
            let reopened = State::open(&model, &cache, None, None).unwrap();
            assert_eq!(
                reopened.model().unwrap()["coverage"]["sha_hashed_shards"],
                0
            );
            assert_eq!(
                reopened.model().unwrap()["coverage"]["sha_verified_shards"],
                0
            );
        }
        siblings[0]["lfs"]["sha256"] = json!("0".repeat(64));
        std::fs::write(
            model.join("model-api.json"),
            json!({"siblings":siblings}).to_string(),
        )
        .unwrap();
        let state = State::open(&model, &root.join("cache-mismatch"), None, None).unwrap();
        assert!(verify_source(&state, 0, Instant::now()).is_err());
        assert_eq!(state.model().unwrap()["coverage"]["sha_hashed_shards"], 0);
        assert!(!state.root.join("verification.json").exists());
        drop(state);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
#[path = "../tests/support/native_command_definition.rs"]
mod native_command_definition_tests;
