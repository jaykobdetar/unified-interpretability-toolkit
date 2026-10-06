//! Private disposable worker protocol. No HTTP, calibration/cache or subprocesses.
use crate::{
    configure, require, sha,
    slice::{parse_indices, TensorSlice},
    source::Source,
    strength::{snapshot, StrengthProfile, CHUNK_VALUES},
    Result,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::fd::{AsRawFd, FromRawFd},
    path::Path,
    time::{Duration, Instant},
};

fn sealed(file: &File, size: usize) -> Result<()> {
    let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    let required = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    require(
        seals >= 0 && seals & required == required && file.metadata()?.len() == size as u64,
        "Snapshot descriptor is not sealed/exact-sized",
    )
}
fn fd(value: &str) -> Result<File> {
    let input: i32 = value.parse()?;
    require(input > 2, "Private inherited FD required")?;
    // Duplicate ownership: never close an arbitrary caller handle via from_raw_fd.
    let owned = unsafe { libc::fcntl(input, libc::F_DUPFD_CLOEXEC, 3) };
    require(owned >= 0, "Cannot duplicate private descriptor")?;
    Ok(unsafe { File::from_raw_fd(owned) })
}
fn check_deadline(end: Instant) -> Result<()> {
    require(Instant::now() < end, "Worker allowance exhausted")
}

pub fn run(options: &BTreeMap<String, String>) -> Result<()> {
    let entered = Instant::now();
    let (deadline, scan_deadline) = admit_options(options, entered)?;
    let source = Source::open(Path::new(&options["model"]))?;
    let slice = TensorSlice::new(
        &source,
        options["tensor"].parse()?,
        &parse_indices(&options["slice"])?,
    )?;
    let model_identity = sha(json!([
        "weight-atlas-model-v1",
        source.identity,
        options["revision"]
    ])
    .to_string()
    .as_bytes());
    let binding = slice.binding(&model_identity).to_string();
    require(
        binding == options["binding"],
        "Worker source/slice correspondence mismatch",
    )?;
    let layout = snapshot::layout(slice.tensor.rows, slice.tensor.cols, binding.len())?;
    limit_output(layout.frame_bytes)?;
    let seed: u64 = options["seed"].parse()?;
    require(seed <= u32::MAX as u64, "Worker seed must fit uint32")?;
    let values: usize = options["values"].parse()?;
    let mut output = candidate_output(options)?;
    let context = ProfileContext {
        source: &source,
        options,
        model_identity: &model_identity,
        seed,
        values,
        scan_deadline,
        frame_bytes: layout.frame_bytes,
    };
    let mut profile = create_profile(&context, slice)?;
    let before = advance_profile(&context, &mut profile)?;
    output.seek(SeekFrom::Start(0))?;
    let revision = profile.write_snapshot(&source, &mut output, deadline)?;
    seal_candidate(&output, layout.frame_bytes)?;
    source.check()?;
    check_deadline(deadline)?;
    // Candidate only. Owner must validate the SAME handle, source and final
    // rusage/reap within its original grant before any terminal publication.
    println!(
        "{}",
        json!({"schema":"weight-atlas.profile-candidate.v1","revision":revision,
        "visited_values":profile.visited_values(),"new_values":profile.visited_values()-before,
        "frame_bytes":layout.frame_bytes,"live_bytes":layout.live_bytes})
    );
    Ok(())
}

struct ProfileContext<'a> {
    source: &'a Source,
    options: &'a BTreeMap<String, String>,
    model_identity: &'a str,
    seed: u64,
    values: usize,
    scan_deadline: Instant,
    frame_bytes: usize,
}

fn admit_options(
    options: &BTreeMap<String, String>,
    entered: Instant,
) -> Result<(Instant, Instant)> {
    let required = [
        "model",
        "revision",
        "tensor",
        "slice",
        "seed",
        "values",
        "wall-ms",
        "cpu-ms",
        "binding",
        "output-fd",
    ];
    require(
        required.iter().all(|k| options.contains_key(*k))
            && options.keys().all(|k| {
                required.contains(&k.as_str()) || ["input-fd", "input-sha"].contains(&k.as_str())
            }),
        "Unknown or missing worker option",
    )?;
    require(
        options.values().map(String::len).sum::<usize>() <= 16384,
        "Worker metadata too large",
    )?;
    require(
        options.contains_key("input-fd") == options.contains_key("input-sha"),
        "Incomplete restore input",
    )?;
    let ms: u64 = options["wall-ms"].parse()?;
    let cpu: u64 = options["cpu-ms"].parse()?;
    require(
        (1..=5000).contains(&ms) && (1..=4000).contains(&cpu),
        "Worker duration outside total grant",
    )?;
    // Parent passes a remainder and independently enforces its ORIGINAL deadline.
    let deadline = entered + Duration::from_millis(ms);
    require(ms > 200, "No finalization margin")?;
    let scan_deadline = deadline - Duration::from_millis(200);
    configure()?;
    let cpu_limit = libc::rlimit {
        rlim_cur: cpu.div_ceil(1000),
        rlim_max: cpu.div_ceil(1000),
    };
    require(
        unsafe { libc::setrlimit(libc::RLIMIT_CPU, &cpu_limit) } == 0,
        "Cannot set child CPU limit",
    )?;
    // RLIMIT_CPU is coarse; aggregate subsecond enforcement belongs to owner.
    check_deadline(scan_deadline)?;
    Ok((deadline, scan_deadline))
}

fn limit_output(frame_bytes: usize) -> Result<()> {
    let output_limit = libc::rlimit {
        rlim_cur: frame_bytes as u64,
        rlim_max: frame_bytes as u64,
    };
    require(
        unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &output_limit) } == 0,
        "Cannot bound child output",
    )?;
    Ok(())
}

fn candidate_output(options: &BTreeMap<String, String>) -> Result<File> {
    let output = fd(&options["output-fd"])?;
    require(
        output.metadata()?.len() == 0,
        "Candidate output must be empty",
    )?;
    let seals = unsafe { libc::fcntl(output.as_raw_fd(), libc::F_GET_SEALS) };
    require(seals == 0, "Candidate must be a new sealable memfd")?;
    Ok(output)
}

fn create_profile(context: &ProfileContext<'_>, slice: TensorSlice) -> Result<StrengthProfile> {
    let profile = if let Some(input) = context.options.get("input-fd") {
        require(
            input != &context.options["output-fd"],
            "Input and output must differ",
        )?;
        let mut input = fd(input)?;
        sealed(&input, context.frame_bytes)?;
        input.seek(SeekFrom::Start(0))?;
        StrengthProfile::restore_snapshot(
            context.source,
            slice,
            context.model_identity,
            context.seed,
            input.take(context.frame_bytes as u64 + 1),
            context.frame_bytes,
            &context.options["input-sha"],
            context.scan_deadline,
        )?
    } else {
        StrengthProfile::new(
            context.source,
            slice,
            context.model_identity,
            context.seed,
            context.values,
            Duration::from_nanos(1),
        )?
    };
    Ok(profile)
}

fn advance_profile(context: &ProfileContext<'_>, profile: &mut StrengthProfile) -> Result<usize> {
    check_deadline(context.scan_deadline)?;
    let before = profile.visited_values();
    profile.authorize(
        context.values,
        context
            .scan_deadline
            .saturating_duration_since(Instant::now()),
    )?;
    while profile.visited_values() - before < context.values
        && Instant::now() < context.scan_deadline
    {
        let previous = profile.visited_values();
        profile.advance_until(
            context.source,
            CHUNK_VALUES.min(context.values - (previous - before)),
            context.scan_deadline,
        )?;
        if profile.visited_values() == previous {
            break;
        }
    }
    Ok(before)
}

fn seal_candidate(output: &File, frame_bytes: usize) -> Result<()> {
    let required_seals =
        libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    require(
        unsafe { libc::fcntl(output.as_raw_fd(), libc::F_ADD_SEALS, required_seals) } == 0,
        "Cannot seal candidate",
    )?;
    sealed(output, frame_bytes)?;
    Ok(())
}
