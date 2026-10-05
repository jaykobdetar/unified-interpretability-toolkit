//! Inactive hosted renderer: one inherited Unix channel, no listener or work queue.
use crate::{
    require, server,
    slice::{parse_indices, TensorSlice},
    state::State,
    Result,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    os::fd::FromRawFd,
    os::unix::net::UnixStream,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    route: String,
    query: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Command {
    version: u8,
    operation_id: String,
    kind: String,
    remaining_ms: u64,
    request: Request,
}

fn execute(state: &State, request: &Request) -> Result<(String, Vec<u8>)> {
    state.source.check()?;
    let allowed: &[&str] = match request.route.as_str() {
        "model" => &[],
        "progress" | "tensor-status" => &["tensor"],
        "binding" => &["tensor", "slice"],
        "view" => &["tensor", "slice", "left", "right"],
        "inspect" => &["tensor", "slice", "row", "col", "left", "right"],
        "tile" => &["tensor", "slice", "rule", "level", "x", "y"],
        "calibration" => &["tensor"],
        _ => return Err("Unknown hosted route".into()),
    };
    require(
        request.query.keys().all(|k| allowed.contains(&k.as_str())),
        "Unknown hosted field",
    )?;
    let q = server::query(
        &request
            .query
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>(),
    );
    let body = match request.route.as_str() {
        "model" => state.model()?,
        "progress" | "tensor-status" => state.status(if request.query.contains_key("tensor") {
            Some(q.int("tensor", "0")?)
        } else {
            None
        })?,
        "inspect" => server::inspect(state, &q)?,
        "binding" | "view" => {
            let id = q.int("tensor", "0")?;
            let slice = TensorSlice::new(&state.source, id, &parse_indices(q.get("slice", ""))?)?;
            if request.route == "binding" {
                json!({"api_version":1,"source_binding":state.slice_binding(&slice)})
            } else {
                let t = state.source.tensor(id)?;
                let mut selected = serde_json::to_value(t)?;
                selected["slice"] = json!(slice.leading);
                selected["slice_identity"] = json!(slice.identity);
                selected["slice_count"] = json!(slice.tensor.count);
                json!({"api_version":1,"tensor":selected,"source_binding":state.slice_binding(&slice),
                    "legends":state.legends(t,q.get("left","global_linear"),q.get("right","global_asinh"))?,
                    "tile_size":256,"overlap":0,"source_values_unchanged":true})
            }
        }
        "calibration" => {
            require(
                request.query.contains_key("tensor"),
                "One explicit tensor required",
            )?;
            state.calibrate_one(q.int("tensor", "0")?)?;
            json!({"api_version":1,"complete":true})
        }
        "tile" => {
            let (png, _, _) = state.tile_slice(
                q.int("tensor", "0")?,
                &parse_indices(q.get("slice", ""))?,
                q.get("rule", "global_linear"),
                q.int("level", "0")?.try_into()?,
                q.int("x", "0")?,
                q.int("y", "0")?,
            )?;
            return Ok(("image/png".into(), png));
        }
        _ => unreachable!(),
    };
    Ok(("application/json".into(), serde_json::to_vec(&body)?))
}

fn channel_io_error(operation: &'static str, error: &std::io::Error) -> String {
    // Static operation and standard ErrorKind/errno only; never include model,
    // cache paths, arbitrary io::Error text, command data or capabilities.
    let errno = error
        .raw_os_error()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".into());
    format!(
        "Hosted channel {operation} failed: kind={:?}; errno={errno}",
        error.kind()
    )
}

pub fn run(state: Arc<State>, fd: i32) -> Result<()> {
    require(fd > 2, "Inherited private channel required")?;
    // Duplicate the owner-provided socket; no path or address is accepted.
    let owned = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    if owned < 0 {
        return Err(channel_io_error("F_DUPFD_CLOEXEC", &std::io::Error::last_os_error()).into());
    }
    let mut stream = unsafe { UnixStream::from_raw_fd(owned) };
    stream
        .peer_addr()
        .map_err(|error| channel_io_error("peer_addr", &error))?;
    loop {
        let mut length = [0u8; 4];
        match stream.read_exact(&mut length) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let length = u32::from_be_bytes(length) as usize;
        require(
            (1..=16384).contains(&length),
            "Hosted command exceeds limit",
        )?;
        stream.set_read_timeout(Some(Duration::from_millis(500)))?;
        let mut raw = vec![0; length];
        stream.read_exact(&mut raw)?;
        let command: Command = serde_json::from_slice(&raw)?;
        require(
            command.version == 1
                && command.operation_id.len() == 32
                && command
                    .operation_id
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                && (1..=5000).contains(&command.remaining_ms),
            "Invalid hosted operation",
        )?;
        let expected = match command.request.route.as_str() {
            "tile" => "tile",
            "calibration" => "calibration",
            "inspect" => "inspect",
            _ => "metadata",
        };
        require(command.kind == expected, "Hosted operation kind mismatch")?;
        let end = Instant::now() + Duration::from_millis(command.remaining_ms);
        // The independent supervisor enforces the original deadline and reaps
        // this renderer if a numerical call cannot return within its allowance.
        let result = execute(&state, &command.request);
        let (status, mime, body) = match result {
            Ok((mime, body)) if Instant::now() < end && body.len() <= 2 * 1024 * 1024 => {
                (200, mime, body)
            }
            _ => (
                400,
                "application/json".into(),
                br#"{"error":"Hosted operation unavailable"}"#.to_vec(),
            ),
        };
        let header = serde_json::to_vec(
            &json!({"ack":{"version":1,"operation_id":command.operation_id,
            "complete":true,"numeric_idle":true},"status":status,"mime":mime,"body_bytes":body.len()}),
        )?;
        let remaining = end
            .checked_duration_since(Instant::now())
            .filter(|v| !v.is_zero())
            .ok_or("Hosted response deadline expired")?;
        stream.set_write_timeout(Some(remaining))?;
        stream.write_all(&(header.len() as u32).to_be_bytes())?;
        stream.write_all(&header)?;
        stream.write_all(&body)?;
        stream.set_read_timeout(None)?;
        // No background calibration, detached jobs, secondary numeric threads,
        // or work continuation after this explicit numeric-idle acknowledgment.
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::channel_io_error;

    #[test]
    fn channel_errno_and_kind_are_preserved_without_syscalls() {
        let error = std::io::Error::from_raw_os_error(libc::EPERM);
        assert_eq!(
            channel_io_error("peer_addr", &error),
            "Hosted channel peer_addr failed: kind=PermissionDenied; errno=1"
        );
        let error = std::io::Error::from_raw_os_error(libc::EBADF);
        assert_eq!(
            channel_io_error("F_DUPFD_CLOEXEC", &error),
            format!(
                "Hosted channel F_DUPFD_CLOEXEC failed: kind={:?}; errno=9",
                error.kind()
            )
        );
    }

    #[test]
    fn custom_error_text_is_never_copied() {
        let error = std::io::Error::other("private-model-path and job-capability");
        assert_eq!(
            channel_io_error("peer_addr", &error),
            "Hosted channel peer_addr failed: kind=Other; errno=none"
        );
    }
}
