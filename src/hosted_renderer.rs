//! Inactive hosted renderer: one inherited Unix channel, no listener or work queue.
use crate::{api, require, server, slice::TensorSlice, state::State, Result};
use serde::Deserialize;
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    os::fd::FromRawFd,
    os::unix::net::UnixStream,
    sync::Arc,
    time::{Duration, Instant},
};

mod response;

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
    let allowed = api::hosted::parameters(&request.route).ok_or("Unknown hosted route")?;
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
        api::hosted::MODEL => state.model()?,
        api::hosted::PROGRESS | api::hosted::TENSOR_STATUS => {
            state.status(if request.query.contains_key(api::parameter::TENSOR) {
                Some(q.read(&api::argument::TENSOR)?)
            } else {
                None
            })?
        }
        api::hosted::INSPECT => server::inspect(state, &q)?,
        api::hosted::BINDING | api::hosted::VIEW => {
            let id = q.read(&api::argument::TENSOR)?;
            let slice = TensorSlice::new(&state.source, id, &q.read(&api::argument::SLICE)?)?;
            if request.route == api::hosted::BINDING {
                Value::from(response::Binding {
                    state,
                    slice: &slice,
                })
            } else {
                let t = state.source.tensor(id)?;
                let selected = Value::from(response::Selection {
                    selected: serde_json::to_value(t)?,
                    slice: &slice,
                });
                Value::try_from(response::View {
                    state,
                    slice: &slice,
                    t,
                    q: &q,
                    selected: &selected,
                })?
            }
        }
        api::hosted::CALIBRATION => {
            require(
                request.query.contains_key(api::parameter::TENSOR),
                "One explicit tensor required",
            )?;
            state.calibrate_one(q.read(&api::argument::TENSOR)?)?;
            Value::from(response::Calibrated)
        }
        api::hosted::TILE => {
            let (png, _, _) = state.tile_slice(
                q.read(&api::argument::TENSOR)?,
                &q.read(&api::argument::SLICE)?,
                q.field(&api::argument::RULE),
                q.read(&api::argument::LEVEL)?,
                q.read(&api::argument::X)?,
                q.read(&api::argument::Y)?,
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
        let header = serde_json::to_vec(&Value::from(response::Frame {
            operation_id: &command.operation_id,
            status,
            mime: &mime,
            body: &body,
        }))?;
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

#[cfg(test)]
mod declaration_vectors {
    use super::*;
    use serde_json::Value;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    struct Fixture {
        state: State,
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "atlas-hosted-declaration-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let model = root.join("model");
            std::fs::create_dir_all(&model).unwrap();
            let header = br#"{"weights":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&[0x80, 0x3f, 0x80, 0xbf]);
            std::fs::write(model.join("tiny.safetensors"), bytes).unwrap();
            let state = State::open(
                &model,
                &root.join("cache"),
                Some("hosted declaration".into()),
                Some("fixture-v1".into()),
            )
            .unwrap();
            Self { state, root }
        }
        fn request(&self, route: &str, query: &[(&str, &str)]) -> Result<(String, Vec<u8>)> {
            execute(
                &self.state,
                &Request {
                    route: route.into(),
                    query: query
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                },
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }
    #[test]
    fn every_private_route_keeps_its_existing_direct_contract() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        for (route, query) in [
            ("model", vec![]),
            ("progress", vec![]),
            ("tensor-status", vec![("tensor", "0")]),
            ("binding", vec![("tensor", "0"), ("slice", "")]),
            ("inspect", vec![("tensor", "0"), ("row", "0"), ("col", "1")]),
            ("calibration", vec![("tensor", "0")]),
            (
                "view",
                vec![
                    ("tensor", "0"),
                    ("slice", ""),
                    ("left", "tensor_linear"),
                    ("right", "tensor_asinh"),
                ],
            ),
        ] {
            let result = fixture.request(route, &query);
            assert!(
                result.is_ok(),
                "hosted declaration route: {route}: {result:?}"
            );
            let (mime, bytes) = result.unwrap();
            assert_eq!(mime, "application/json", "hosted declaration mime: {route}");
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["api_version"], 1, "hosted declaration JSON: {route}");
            if route == "model" {
                assert_eq!(
                    body["name"], "hosted declaration",
                    "hosted model declaration"
                );
            }
            if route == "binding" {
                assert_eq!(
                    body.as_object().unwrap().len(),
                    2,
                    "hosted binding declaration"
                );
            }
            if route == "inspect" {
                assert_eq!(body["raw_exact"], "-1", "hosted inspect declaration");
            }
        }
        let (mime, png) = fixture
            .request(
                "tile",
                &[
                    ("tensor", "0"),
                    ("rule", "tensor_linear"),
                    ("level", "1"),
                    ("x", "0"),
                    ("y", "0"),
                ],
            )
            .unwrap();
        assert_eq!(mime, "image/png", "hosted tile declaration");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(png.len() < 64 * 1024);
    }
    #[test]
    fn private_admission_keeps_exact_unknown_missing_and_parse_order() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        for (route, query, message) in [
            ("Model", vec![], "Unknown hosted route"),
            ("model", vec![("unexpected", "1")], "Unknown hosted field"),
            (
                "binding",
                vec![("tensor", "bad"), ("unexpected", "1")],
                "Unknown hosted field",
            ),
            ("calibration", vec![], "One explicit tensor required"),
            (
                "binding",
                vec![("tensor", "bad")],
                "invalid digit found in string",
            ),
        ] {
            assert_eq!(
                fixture.request(route, &query).unwrap_err().to_string(),
                message,
                "hosted declaration refusal: {route}"
            );
        }
    }

    // Additional hosted response witnesses; original declaration tests above are unchanged.
    #[test]
    fn hosted_binding_view_and_calibration_keep_exact_envelopes() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (mime, bytes) = fixture.request("calibration", &[("tensor", "0")]).unwrap();
        assert_eq!(mime, "application/json");
        assert_eq!(bytes, br#"{"api_version":1,"complete":true}"#);
        let slice = TensorSlice::new(&fixture.state.source, 0, &[]).unwrap();
        let binding = fixture.state.slice_binding(&slice);
        let (_, bytes) = fixture
            .request("binding", &[("tensor", "0"), ("slice", "")])
            .unwrap();
        assert_eq!(
            bytes,
            serde_json::to_vec(&json!({"api_version":1,"source_binding":binding})).unwrap()
        );
        let (_, bytes) = fixture
            .request(
                "view",
                &[
                    ("tensor", "0"),
                    ("slice", ""),
                    ("left", "tensor_linear"),
                    ("right", "tensor_asinh"),
                ],
            )
            .unwrap();
        let tensor = &fixture.state.source.tensors[0];
        let mut selected = serde_json::to_value(tensor).unwrap();
        selected["slice"] = json!([]);
        selected["slice_identity"] = json!(slice.identity);
        selected["slice_count"] = json!(2);
        let legends = fixture
            .state
            .legends(tensor, "tensor_linear", "tensor_asinh")
            .unwrap();
        let expected = json!({"api_version":1,"tensor":selected,"source_binding":binding,"legends":legends,
            "tile_size":256,"overlap":0,"source_values_unchanged":true});
        assert_eq!(bytes, serde_json::to_vec(&expected).unwrap());
    }

    struct LocalChannel {
        stream: UnixStream,
        worker: Option<std::thread::JoinHandle<Result<()>>>,
    }
    impl LocalChannel {
        fn new(fixture: &Fixture) -> Self {
            let state = Arc::new(
                State::open(
                    &fixture.root.join("model"),
                    &fixture.root.join("wire-cache"),
                    None,
                    None,
                )
                .unwrap(),
            );
            let (stream, receiver) = UnixStream::pair().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let worker = std::thread::spawn(move || {
                let fd = std::os::fd::AsRawFd::as_raw_fd(&receiver);
                run(state, fd)
            });
            Self {
                stream,
                worker: Some(worker),
            }
        }
    }
    impl Drop for LocalChannel {
        fn drop(&mut self) {
            let _ = self.stream.shutdown(std::net::Shutdown::Both);
            let result = self.worker.take().unwrap().join();
            if !std::thread::panicking() {
                assert!(
                    matches!(result, Ok(Ok(()))),
                    "owned hosted thread cleanup assertion: {result:?}"
                );
            }
        }
    }

    #[test]
    fn private_channel_acknowledges_exact_completed_body_and_reaps() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let mut channel = LocalChannel::new(&fixture);
        let id = "0123456789abcdef0123456789abcdef";
        let command =
            serde_json::to_vec(&json!({"version":1,"operation_id":id,"kind":"calibration",
            "remaining_ms":5000,"request":{"route":"calibration","query":{"tensor":"0"}}}))
            .unwrap();
        channel
            .stream
            .write_all(&(command.len() as u32).to_be_bytes())
            .unwrap();
        channel.stream.write_all(&command).unwrap();
        let mut size = [0u8; 4];
        let read = channel.stream.read_exact(&mut size);
        assert!(
            read.is_ok(),
            "bounded header prefix read assertion: {read:?}"
        );
        let size = u32::from_be_bytes(size) as usize;
        assert!(size <= 4096, "header length bound assertion");
        let mut header = vec![0u8; size];
        let read = channel.stream.read_exact(&mut header);
        assert!(read.is_ok(), "bounded header read assertion: {read:?}");
        let body = br#"{"api_version":1,"complete":true}"#;
        let expected = json!({"ack":{"version":1,"operation_id":id,"complete":true,"numeric_idle":true},
            "status":200,"mime":"application/json","body_bytes":body.len()});
        assert_eq!(
            header,
            serde_json::to_vec(&expected).unwrap(),
            "exact private acknowledgement bytes"
        );
        let mut received = vec![0u8; body.len()];
        let read = channel.stream.read_exact(&mut received);
        assert!(read.is_ok(), "bounded body read assertion: {read:?}");
        assert_eq!(received, body);
        drop(channel);
        // The owned thread joins before the fixture removes its source/cache tree.
    }
}
