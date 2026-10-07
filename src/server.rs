mod response;
mod reuse;
mod reuse_transport;
use crate::{
    api, headroom, render, require,
    slice::{parse_indices, TensorSlice},
    source::{exact_decimal_for, Dtype},
    state::State,
    Result,
};
use reuse_transport::Connection;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    os::fd::AsRawFd,
    sync::{
        mpsc::{sync_channel, SyncSender, TryRecvError},
        Arc,
    },
    time::{Duration, Instant},
};
#[derive(Clone)]
pub struct Query(BTreeMap<String, String>);
impl Query {
    pub fn get<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.0.get(key).map(String::as_str).unwrap_or(default)
    }
    pub fn int(&self, key: &str, default: &str) -> Result<usize> {
        Ok(self.get(key, default).parse()?)
    }
}
fn decode(s: &str) -> Result<String> {
    fn hex(b: u8) -> Result<u8> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err("Invalid URL escape".into()),
        }
    }
    let mut v = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            require(i + 2 < b.len(), "Invalid URL escape")?;
            v.push((hex(b[i + 1])? << 4) | hex(b[i + 2])?);
            i += 3
        } else {
            v.push(if b[i] == b'+' { b' ' } else { b[i] });
            i += 1
        }
    }
    Ok(String::from_utf8(v)?)
}
fn parse_query(s: &str) -> Result<Query> {
    let mut m = BTreeMap::new();
    for p in s.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = p.split_once('=').ok_or("Invalid query")?;
        require(
            m.insert(decode(k)?, decode(v)?).is_none(),
            "Duplicate query parameter",
        )?
    }
    Ok(Query(m))
}
const RESPONSE_DEADLINE: Duration = Duration::from_secs(3);

trait TimedWrite: Write {
    fn write_timeout(&self, timeout: Duration) -> io::Result<()>;
}
impl TimedWrite for TcpStream {
    fn write_timeout(&self, timeout: Duration) -> io::Result<()> {
        self.set_write_timeout(Some(timeout))
    }
}
fn write_response(
    writer: &mut impl TimedWrite,
    parts: &[&[u8]],
    deadline: Instant,
    mut now: impl FnMut() -> Instant,
) -> io::Result<()> {
    for part in parts {
        let mut remaining = *part;
        while !remaining.is_empty() {
            let budget = deadline
                .checked_duration_since(now())
                .filter(|d| !d.is_zero())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::TimedOut, "Response deadline exceeded")
                })?;
            writer.write_timeout(budget)?;
            match writer.write(remaining) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => {
                    remaining = &remaining[n..];
                    if now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "Response deadline exceeded",
                        ));
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}
fn response_head(status: u16, mime: &str, length: usize, headers: &str) -> String {
    response_head_with_connection(status, mime, length, headers, false)
}
fn response_head_with_connection(
    status: u16,
    mime: &str,
    length: usize,
    headers: &str,
    reusable: bool,
) -> String {
    let connection = if reusable { "keep-alive" } else { "close" };
    let phrase = match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Error",
    };
    let cache = if headers.contains("Cache-Control:") {
        ""
    } else {
        "Cache-Control: no-store\r\n"
    };
    format!("HTTP/1.1 {status} {phrase}\r\nContent-Type: {mime}\r\nContent-Length: {length}\r\nConnection: {connection}\r\n{cache}X-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'\r\n{headers}\r\n")
}
pub fn reply(mut socket: TcpStream, status: u16, mime: &str, body: &[u8], headers: &str) {
    let text = response_head(status, mime, body.len(), headers);
    if socket.set_nonblocking(false).is_ok() {
        let _ = write_response(
            &mut socket,
            &[text.as_bytes(), body],
            Instant::now() + RESPONSE_DEADLINE,
            Instant::now,
        );
    }
}
// One fixed startup resource keeps the normal page below the unchanged four
// header-admission slots. Preserve script order without eval, loaders or copies.
fn viewer_scripts() -> [&'static [u8]; 9] {
    [
        include_bytes!("../web/vendor/openseadragon.min.js"),
        b"\n;\n",
        include_bytes!("../web/atlas-tools.js"),
        b"\n;\n",
        include_bytes!("../web/app.js"),
        b"\n;\n",
        include_bytes!("../web/workspace-tools.js"),
        b"\n;\n",
        include_bytes!("../web/inference.js"),
    ]
}
fn reply_viewer_bundle(mut socket: TcpStream) {
    let scripts = viewer_scripts();
    let head = response_head(
        200,
        "text/javascript",
        scripts.iter().map(|s| s.len()).sum(),
        "",
    );
    let mut parts = Vec::with_capacity(scripts.len() + 1);
    parts.push(head.as_bytes());
    parts.extend_from_slice(&scripts);
    if socket.set_nonblocking(false).is_ok() {
        let _ = write_response(
            &mut socket,
            &parts,
            Instant::now() + RESPONSE_DEADLINE,
            Instant::now,
        );
    }
}
// Intake never waits for output: one best-effort write, then close. Full responses
// are owned by the bounded dispatch/numeric workers, away from header progress.
fn reject_now(mut socket: TcpStream, status: u16, code: &str, message: &str) {
    let body = json!({"api_version":1,"code":code,"error":message}).to_string();
    let wire = response_head(status, "application/json", body.len(), "") + &body;
    if socket.set_nonblocking(true).is_ok() {
        let _ = socket.write(wire.as_bytes());
    }
}
pub(crate) fn json_reply(s: TcpStream, status: u16, v: Value) {
    reply(s, status, "application/json", v.to_string().as_bytes(), "")
}
pub(crate) fn error(s: TcpStream, status: u16, e: impl std::fmt::Display) {
    json_reply(s, status, json!({"error":e.to_string(),"api_version":1}))
}
enum Job {
    Wake,
    Calibrate(usize),
    Tile(Connection, Query),
}
pub(crate) fn disconnected(s: &TcpStream) -> bool {
    let _ = s.set_nonblocking(true);
    let r = s.peek(&mut [0u8; 1]);
    let _ = s.set_nonblocking(false);
    matches!(r, Ok(0))
}
pub fn inspect(state: &State, q: &Query) -> Result<Value> {
    let id = q.int("tensor", "0")?;
    let slice = TensorSlice::new(&state.source, id, &parse_indices(q.get("slice", ""))?)?;
    let t = &slice.tensor;
    let row = q.int("row", "0")?;
    let col = q.int("col", "0")?;
    let (bits, offset) = state.source.scalar_bits(t, row, col)?;
    let dtype = Dtype::parse(&t.dtype)?;
    let value = dtype.value(bits);
    let mut fields = serde_json::Map::new();
    let mut transform_errors = serde_json::Map::new();
    let mut ready = true;
    for (side, rule) in [
        ("left", q.get("left", "global_linear")),
        ("right", q.get("right", "global_asinh")),
    ] {
        render::rule_info(rule)?;
        let result = (|| -> Result<Value> {
            render::validate_rule_dtype(rule, dtype)?;
            require(
                value.is_finite(),
                "Nonfinite source value: transform unavailable",
            )?;
            let legend = render::legend(rule, state.stats(id).as_ref(), state.global())?;
            Ok(json!(state.tensor_mapping(t, rule, &legend)?.value(bits)))
        })();
        let field = match result {
            Ok(value) => value,
            Err(error) => {
                ready = false;
                transform_errors.insert(side.into(), json!(error.to_string()));
                Value::Null
            }
        };
        fields.insert(side.into(), field);
    }
    let raw = bits.to_le_bytes();
    let raw_hex = raw[..dtype.bytes()]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let classification = if value.is_nan() {
        "nan"
    } else if value == f64::INFINITY {
        "positive_infinity"
    } else if value == f64::NEG_INFINITY {
        "negative_infinity"
    } else {
        "finite"
    };
    Ok(response::Inspection {
        source_binding: state.slice_binding(&slice),
        tensor: id,
        row,
        col,
        raw_exact: exact_decimal_for(dtype, bits),
        dtype: dtype.name(),
        element_bytes: dtype.bytes(),
        raw_hex_le: &raw_hex,
        classification,
        bf16_hex_le: (dtype == Dtype::Bf16).then_some(raw_hex.as_str()),
        shard: &t.shard,
        byte_offset: offset,
        native_indices: slice.native_indices(row, col)?,
        transformed: fields,
        transform_errors,
        transforms_ready: ready,
    }
    .into())
}
type Request = (String, String, Query, BTreeMap<String, String>);
const MAX_PENDING_HEADERS: usize = 4;
const HEADER_LIMIT: usize = 8192;
const HEADER_DEADLINE: Duration = Duration::from_millis(500);

struct HeaderReader {
    deadline: Instant,
    raw: Vec<u8>,
}
impl HeaderReader {
    fn new(now: Instant) -> Self {
        Self {
            deadline: now + HEADER_DEADLINE,
            raw: Vec::new(),
        }
    }
    // Nonblocking reader; fake readers/clocks exercise deadlines without a server.
    fn poll(
        &mut self,
        reader: &mut impl Read,
        mut now: impl FnMut() -> Instant,
    ) -> Result<Option<Vec<u8>>> {
        for _ in 0..4 {
            require(
                now() < self.deadline,
                "HTTP request receive deadline exceeded",
            )?;
            let mut bytes = [0; 1024];
            let n = match reader.read(&mut bytes) {
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            };
            require(n > 0, "Empty HTTP request")?;
            require(
                now() < self.deadline,
                "HTTP request receive deadline exceeded",
            )?;
            require(
                self.raw.len() + n <= HEADER_LIMIT,
                "HTTP headers exceed 8 KiB",
            )?;
            self.raw.extend_from_slice(&bytes[..n]);
            if self.raw.windows(4).any(|w| w == b"\r\n\r\n") {
                return Ok(Some(std::mem::take(&mut self.raw)));
            }
        }
        Ok(None)
    }
}
struct PendingHeader<T = TcpStream> {
    socket: T,
    reader: HeaderReader,
}
fn collect_headers<T: Read>(
    pending: &mut Vec<PendingHeader<T>>,
    mut now: impl FnMut() -> Instant,
) -> Vec<(T, Result<Vec<u8>>)> {
    let mut completed = Vec::new();
    let mut i = 0;
    while i < pending.len() {
        let p = &mut pending[i];
        match p.reader.poll(&mut p.socket, &mut now) {
            Ok(None) => i += 1,
            result => {
                let p = pending.swap_remove(i);
                completed.push((p.socket, result.map(|raw| raw.expect("completed header"))));
            }
        }
    }
    completed
}
pub(crate) type CompletedRequest = (TcpStream, Vec<u8>);
pub(crate) const DISPATCH_CAPACITY: usize = 4;
fn recoverable_accept(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::Interrupted
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) || matches!(
        error.raw_os_error(),
        Some(
            libc::EMFILE
                | libc::ENFILE
                | libc::ENOBUFS
                | libc::ENOMEM
                | libc::ENETDOWN
                | libc::EPROTO
                | libc::EHOSTUNREACH
                | libc::ENETUNREACH
        )
    )
}
fn accept_backoff(failures: u32) -> Duration {
    Duration::from_millis((10u64 << failures.min(6)).min(250))
}

pub fn serve(state: Arc<State>, port: u16) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let port = listener.local_addr()?.port();
    let reuse = reuse_transport::Runtime::new().ok();
    let worker_reuse = reuse.clone();
    let (sender, receiver) = sync_channel::<Job>(8);
    let worker = state.clone();
    std::thread::Builder::new().name("atlas-numeric-worker".into()).spawn(move || loop {
        let job = match receiver.try_recv() {
            Ok(job) => Some(job),
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => None,
        };
        let job = job.or_else(|| {
            let all = worker.progress.lock().unwrap().all_requested;
            let next = if all {
                worker.source.tensors.iter().find(|t| t.available && worker.stats(t.id).is_none()).map(|t| t.id)
            } else { None };
            if let Some(id) = next { Some(Job::Calibrate(id)) }
            else {
                if all { worker.progress.lock().unwrap().all_requested = false; }
                // OS channel wait; explicit calibration/tile sends wake the worker.
                receiver.recv().ok()
            }
        });
        match job {
            Some(Job::Wake) => {},
            Some(Job::Calibrate(id)) => {
                if let Err(error) = worker.calibrate_one(id) {
                    eprintln!("Calibration paused: {error}");
                    worker.progress.lock().unwrap().all_requested = false;
                }
            }
            Some(Job::Tile(connection, q)) => {
                if disconnected(&connection.socket) { continue; }
                if connection.expired(Instant::now()) {
                    connection.close_error(503, "Connection reuse eligibility expired");
                    continue;
                }
                let start = Instant::now();
                let result = (|| {
                    let binding = q.get("binding", "");
                    worker.tile_slice_bound(q.int("tensor", "0")?, &parse_indices(q.get("slice", ""))?,
                        q.get("rule", "global_linear"), q.int("level", "0")?.try_into()?,
                        q.int("x", "0")?, q.int("y", "0")?, (!binding.is_empty()).then_some(binding))
                })();
                match result {
                    Ok((png, cached, metrics)) => {
                        let mut headers = format!("X-Atlas-Factor: {}\r\nX-Atlas-Cache: {}\r\nX-Atlas-Seconds: {:.6}\r\nX-Atlas-Source-Bytes: {}\r\n",
                            metrics.factor, if cached { "hit" } else { "miss" }, start.elapsed().as_secs_f64(), metrics.source_bytes_read);
                        if !q.get("binding", "").is_empty() {
                            headers.push_str("Cache-Control: private, max-age=86400, immutable\r\n");
                        }
                        reuse_transport::reply_tile(connection, &png, &headers, worker_reuse.as_deref());
                    }
                    Err(error_value) => {
                        let code = if error_value.to_string().contains("not ready") { 503 } else { 400 };
                        connection.close_error(code, error_value);
                    }
                }
            }
            None => break,
        }
    })?;
    let (dispatch_sender, dispatch_receiver) =
        sync_channel::<reuse_transport::Completed>(DISPATCH_CAPACITY);
    let dispatch_state = state.clone();
    let numeric_sender = sender.clone();
    let _dispatch_thread = std::thread::Builder::new()
        .name("atlas-dispatch".into())
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            for (socket, raw) in dispatch_receiver {
                dispatch(&dispatch_state, &numeric_sender, socket, &raw, port);
            }
        })?;
    println!(
        "{}",
        json!({"listening":format!("http://127.0.0.1:{port}"),"metadata_ready_seconds":state.started.elapsed().as_secs_f64(),"header_bytes":state.source.header_bytes,"tensors":state.source.tensors.len(),"peak_rss_mib":crate::peak_rss_mib(),"resources":crate::resources::snapshot()?})
    );
    std::io::stdout().flush()?;
    reuse_transport::run(listener, dispatch_sender, reuse)
}

// Shared bounded intake; comparison uses the same admission and header deadlines.
pub(crate) fn run_transport(
    listener: TcpListener,
    dispatch_sender: SyncSender<CompletedRequest>,
) -> Result<()> {
    listener.set_nonblocking(true)?;
    let mut pending: Vec<PendingHeader> = Vec::with_capacity(MAX_PENDING_HEADERS);
    let mut next_accept = Instant::now();
    let mut failures = 0u32;
    loop {
        if Instant::now() >= next_accept {
            // Bound admission work even when new connections are continuously ready.
            for _ in 0..MAX_PENDING_HEADERS {
                match listener.accept() {
                    Ok((socket, _)) => {
                        failures = 0;
                        if socket.set_nonblocking(true).is_err() {
                            continue;
                        }
                        if pending.len() == MAX_PENDING_HEADERS {
                            // Best-effort reply on a nonblocking socket; never wait for admission.
                            reject_now(
                                socket,
                                503,
                                "admission_full",
                                "Request admission full; retry shortly",
                            );
                        } else {
                            pending.push(PendingHeader {
                                socket,
                                reader: HeaderReader::new(Instant::now()),
                            });
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) if recoverable_accept(&e) => {
                        failures = failures.saturating_add(1).min(6);
                        next_accept = Instant::now() + accept_backoff(failures);
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }
        // This thread only progresses headers and bounded channel admission.
        // Metadata/filesystem work and response writes cannot consume its clock.
        for (socket, result) in collect_headers(&mut pending, Instant::now) {
            match result {
                Ok(raw) => {
                    if let Err(e) = dispatch_sender.try_send((socket, raw)) {
                        let (std::sync::mpsc::TrySendError::Full((socket, _))
                        | std::sync::mpsc::TrySendError::Disconnected((socket, _))) = e;
                        reject_now(
                            socket,
                            503,
                            "dispatch_full",
                            "Request dispatch full; retry shortly",
                        );
                    }
                }
                Err(e) => reject_now(socket, 400, "request_input", &e.to_string()),
            }
        }
        wait_transport(&listener, &pending, next_accept)?;
    }
}

// Wait for socket readiness or the nearest existing absolute deadline. No idle
// timer and no extension of a header/accept deadline when readiness is spurious.
fn poll_timeout(now: Instant, deadlines: impl Iterator<Item = Instant>) -> i32 {
    deadlines
        .min()
        .map(|deadline| {
            let remaining = deadline.saturating_duration_since(now);
            remaining
                .as_nanos()
                .div_ceil(1_000_000)
                .min(i32::MAX as u128) as i32
        })
        .unwrap_or(-1)
}
fn wait_transport(listener: &TcpListener, pending: &[PendingHeader], retry: Instant) -> Result<()> {
    let now = Instant::now();
    let retrying = retry > now;
    let mut descriptors = Vec::with_capacity(pending.len() + 1);
    if !retrying {
        descriptors.push(libc::pollfd {
            fd: listener.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });
    }
    descriptors.extend(pending.iter().map(|p| libc::pollfd {
        fd: p.socket.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    }));
    let timeout = poll_timeout(
        now,
        pending
            .iter()
            .map(|p| p.reader.deadline)
            .chain(retrying.then_some(retry)),
    );
    // SAFETY: all descriptors are owned for the entire call; poll writes only
    // inside the initialized vector. EINTR returns to recheck absolute clocks.
    let result = unsafe {
        libc::poll(
            descriptors.as_mut_ptr(),
            descriptors.len() as libc::nfds_t,
            timeout,
        )
    };
    if result < 0 {
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error.into());
        }
    }
    Ok(())
}

pub(crate) fn parse_request(raw: &[u8], port: u16) -> Result<Request> {
    let text = std::str::from_utf8(raw)?;
    let mut lines = text.split("\r\n");
    let mut first = lines.next().ok_or("Missing request")?.split_whitespace();
    let method = first.next().ok_or("Missing method")?.to_owned();
    let url = first.next().ok_or("Missing URL")?;
    require(
        first.next() == Some("HTTP/1.1") || text.starts_with("GET ") && text.contains("HTTP/1.0"),
        "Unsupported HTTP version",
    )?;
    require(
        url.starts_with('/') && !url.starts_with("//"),
        "Only local paths accepted",
    )?;
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let mut headers = BTreeMap::new();
    for line in lines.take_while(|s| !s.is_empty()) {
        let (k, v) = line.split_once(':').ok_or("Invalid HTTP header")?;
        require(
            headers
                .insert(k.to_ascii_lowercase(), v.trim().to_owned())
                .is_none(),
            "Duplicate HTTP header",
        )?;
    }
    let host = headers.get("host").map(String::as_str).unwrap_or("");
    require(
        host == format!("127.0.0.1:{port}") || host == format!("localhost:{port}"),
        "Host must be loopback",
    )?;
    if let Some(origin) = headers.get("origin") {
        require(
            origin == &format!("http://{host}"),
            "Cross-origin requests are not accepted",
        )?
    }
    if let Some(site) = headers.get("sec-fetch-site") {
        require(site != "cross-site", "Cross-site requests are not accepted")?
    }
    require(
        headers.get("content-length").is_none_or(|s| s == "0")
            && !headers.contains_key("transfer-encoding"),
        "Request bodies are unsupported",
    )?;
    Ok((method, path.into(), parse_query(query)?, headers))
}

fn dispatch(
    state: &Arc<State>,
    sender: &SyncSender<Job>,
    mut connection: Connection,
    raw: &[u8],
    port: u16,
) {
    if connection.expired(Instant::now()) {
        connection.close_error(503, "Connection reuse eligibility expired");
        return;
    }
    if connection.lease.is_some() && !reuse::bounded_framing(raw) {
        connection.close_error(
            400,
            "One bounded HTTP header required on a reused connection",
        );
        return;
    }
    let request = parse_request(raw, port).and_then(|request| {
        headroom()?;
        Ok(request)
    });

    let (method, path, q, headers) = match request {
        Ok(v) => v,
        Err(e) => {
            connection.close_error(400, e);
            return;
        }
    };
    connection.eligible = reuse::candidate(
        raw,
        &method,
        &path,
        q.get(api::parameter::BINDING, ""),
        headers.get("connection").map(String::as_str).unwrap_or(""),
    );
    let Connection {
        socket,
        lease,
        eligible,
    } = connection;
    if method == "POST" && path == api::viewer::CALIBRATION {
        reply_calibration(state, sender, socket, &q, &headers);
        return;
    }
    if method != "GET" {
        error(
            socket,
            400,
            "Only GET and local calibration POST are supported",
        );
        return;
    }
    if path == api::viewer::TILE {
        enqueue_tile(
            sender,
            Connection {
                socket,
                lease,
                eligible,
            },
            q,
        );
        return;
    }
    if path == api::viewer::PROGRESS || path == api::viewer::TENSOR_STATUS {
        reply_status(state, socket, &path, &q);
        return;
    }
    if path == api::viewer::MODEL {
        match state.model() {
            Ok(v) => json_reply(socket, 200, v),
            Err(e) => error(socket, 400, e),
        }
        return;
    }
    if path == api::viewer::INSPECT {
        match inspect(state, &q) {
            Ok(v) => json_reply(socket, 200, v),
            Err(e) => error(socket, 400, e),
        }
        return;
    }
    if path == api::viewer::VIEW {
        reply_view(state, socket, &q);
        return;
    }
    if path == "/viewer.js" {
        reply_viewer_bundle(socket);
        return;
    }
    reply_static(socket, &path)
}

// Queue admission and clearing the previous error share the publication lock.
// The private callback must remain nonblocking: the worker also needs this lock.
fn queue_calibration(state: &State, enqueue: impl FnOnce() -> bool) -> bool {
    let mut progress = state.progress.lock().unwrap();
    if !enqueue() {
        return false;
    }
    progress.error = None;
    true
}

fn reply_calibration(
    state: &State,
    sender: &SyncSender<Job>,
    socket: TcpStream,
    q: &Query,
    headers: &BTreeMap<String, String>,
) {
    if headers.get("x-atlas-local").map(String::as_str) != Some("1") {
        error(socket, 400, "Local action header required");
        return;
    }
    if q.get("all", "0") == "1" {
        let mut p = state.progress.lock().unwrap();
        // Wake a sleeping worker without recurring polling. A full queue
        // already wakes it; disconnected transport cannot accept work.
        if matches!(
            sender.try_send(Job::Wake),
            Err(std::sync::mpsc::TrySendError::Disconnected(_))
        ) {
            error(socket, 503, "Numeric worker unavailable");
            return;
        }
        p.all_requested = true;
        p.error = None;
        json_reply(
            socket,
            202,
            json!({"api_version":1,"queued":if state.source.tensors.iter().all(|t|t.available){"complete checkpoint"}else{"supported tensors; global calibration unavailable"}}),
        );
        return;
    }
    match q.int("tensor", "0").and_then(|id| {
        let t = state.source.tensor(id)?;
        require(
            t.available,
            t.unavailable_reason
                .as_deref()
                .unwrap_or("Tensor unavailable"),
        )?;
        Ok(id)
    }) {
        Ok(id) => {
            if state.stats(id).is_some() {
                json_reply(socket, 200, json!({"api_version":1,"complete":true}));
            } else if queue_calibration(state, || sender.try_send(Job::Calibrate(id)).is_ok()) {
                json_reply(socket, 202, json!({"api_version":1,"queued":id}));
            } else {
                error(socket, 503, "Numeric queue full; retry shortly")
            }
        }
        Err(e) => error(socket, 400, e),
    }
}

fn enqueue_tile(sender: &SyncSender<Job>, connection: Connection, q: Query) {
    match sender.try_send(Job::Tile(connection, q)) {
        Ok(()) => {}
        Err(e) => {
            let (std::sync::mpsc::TrySendError::Full(job)
            | std::sync::mpsc::TrySendError::Disconnected(job)) = e;
            if let Job::Tile(s, _) = job {
                s.close_error(503, "Numeric queue full; retry shortly")
            }
        }
    }
}

fn reply_status(state: &State, socket: TcpStream, path: &str, q: &Query) {
    let result = (|| -> Result<Value> {
        let selected = if path == api::viewer::TENSOR_STATUS
            || !q.get(api::parameter::TENSOR, "").is_empty()
        {
            require(
                !q.get(api::parameter::TENSOR, "").is_empty(),
                "Selected tensor required",
            )?;
            Some(q.int(api::parameter::TENSOR, "0")?)
        } else {
            None
        };
        state.status(selected)
    })();
    match result {
        Ok(value) => json_reply(socket, 200, value),
        Err(e) => error(socket, 400, e),
    }
}

fn reply_view(state: &State, socket: TcpStream, q: &Query) {
    let result = (|| -> Result<Value> {
        state.source.check()?;
        let id = q.int("tensor", "0")?;
        let slice = TensorSlice::new(&state.source, id, &parse_indices(q.get("slice", ""))?)?;
        let t = state.source.tensor(id)?;
        let mut selected = serde_json::to_value(t)?;
        selected["slice"] = json!(slice.leading);
        selected["slice_identity"] = json!(slice.identity);
        selected["slice_count"] = json!(slice.tensor.count);
        let l = state.legends(
            t,
            q.get("left", "global_linear"),
            q.get("right", "global_asinh"),
        )?;
        Ok(response::View {
            tensor: selected,
            source_binding: state.slice_binding(&slice),
            left_binding: state.tile_binding_for(
                &slice,
                q.get("left", "global_linear"),
                &l["left"],
            ),
            right_binding: state.tile_binding_for(
                &slice,
                q.get("right", "global_asinh"),
                &l["right"],
            ),
            legends: l,
        }
        .into())
    })();
    match result {
        Ok(v) => json_reply(socket, 200, v),
        Err(e) => {
            let code = if e.to_string().contains("not ready") {
                503
            } else {
                400
            };
            error(socket, code, e)
        }
    }
}

fn reply_static(socket: TcpStream, path: &str) {
    let static_file: Option<(&str, &[u8])> = match path {
        "/" | "/index.html" => Some((
            "text/html; charset=utf-8",
            include_bytes!("../web/index.html"),
        )),
        "/inference.js" => Some(("text/javascript", include_bytes!("../web/inference.js"))),
        "/inference-import.js" => Some((
            "text/javascript",
            include_bytes!("../web/inference-import.js"),
        )),
        "/atlas-tools.js" => Some(("text/javascript", include_bytes!("../web/atlas-tools.js"))),
        "/workspace-tools.js" => Some((
            "text/javascript",
            include_bytes!("../web/workspace-tools.js"),
        )),
        "/app.js" => Some(("text/javascript", include_bytes!("../web/app.js"))),
        "/style.css" => Some(("text/css", include_bytes!("../web/style.css"))),
        "/vendor/openseadragon.min.js" => Some((
            "text/javascript",
            include_bytes!("../web/vendor/openseadragon.min.js"),
        )),
        "/vendor/OpenSeadragon-LICENSE.txt" => Some((
            "text/plain",
            include_bytes!("../web/vendor/OpenSeadragon-LICENSE.txt"),
        )),
        _ => None,
    };
    if let Some((mime, body)) = static_file {
        reply(socket, 200, mime, body, "")
    } else {
        error(socket, 404, "Not found")
    }
}

pub fn query(args: &[(String, String)]) -> Query {
    Query(args.iter().cloned().collect())
}

#[cfg(test)]
mod query_tests {
    use super::*;

    #[test]
    fn immutable_cache_headers_require_explicit_validated_tile_policy() {
        let legacy = response_head(200, "image/png", 10, "");
        assert!(legacy.contains("Cache-Control: no-store\r\n"));
        assert!(!legacy.contains("immutable"));
        let bound = response_head(
            200,
            "image/png",
            10,
            "Cache-Control: private, max-age=86400, immutable\r\n",
        );
        assert_eq!(bound.matches("Cache-Control:").count(), 1);
        assert!(bound.contains("private, max-age=86400, immutable"));
    }

    #[test]
    fn os_wait_uses_the_earliest_absolute_deadline() {
        let now = Instant::now();
        assert_eq!(poll_timeout(now, std::iter::empty()), -1);
        assert_eq!(
            poll_timeout(
                now,
                [
                    now + Duration::from_millis(500),
                    now + Duration::from_millis(12)
                ]
                .into_iter()
            ),
            12
        );
        assert_eq!(
            poll_timeout(now, [now + Duration::from_micros(1)].into_iter()),
            1
        );
        assert_eq!(poll_timeout(now, [now].into_iter()), 0);
    }

    #[test]
    fn startup_script_bundle_is_fixed_order_bounded_and_single_request() {
        let parts = viewer_scripts();
        assert!(parts.iter().map(|s| s.len()).sum::<usize>() < 600 * 1024);
        assert_eq!(
            parts[0],
            include_bytes!("../web/vendor/openseadragon.min.js")
        );
        assert_eq!(parts[2], include_bytes!("../web/atlas-tools.js"));
        assert_eq!(parts[4], include_bytes!("../web/app.js"));
        assert_eq!(parts[6], include_bytes!("../web/workspace-tools.js"));
        assert_eq!(parts[8], include_bytes!("../web/inference.js"));
        for part in [parts[1], parts[3], parts[5], parts[7]] {
            assert_eq!(part, b"\n;\n");
        }
        let html = include_str!("../web/index.html");
        assert_eq!(html.matches("<script ").count(), 1);
        assert!(html.contains("<script defer src=\"./viewer.js\"></script>"));
    }

    #[test]
    fn query_unicode_and_form_encoding_remain_compatible() {
        assert_eq!(decode("café + tea").unwrap(), "café   tea");
        assert_eq!(decode("caf%C3%A9%20%2b%20tea").unwrap(), "café + tea");
        assert_eq!(decode("%E2%98%83").unwrap(), "☃");
        assert!(parse_query("name=a&%6eame=b").is_err());
        assert_eq!(
            parse_query("name=caf%C3%A9").unwrap().get("name", ""),
            "café"
        );
    }

    #[test]
    fn invalid_query_encoding_returns_errors_in_the_patched_helper() {
        for text in ["%", "%0", "%gg", "%é", "%0é", "%ff"] {
            assert!(decode(text).is_err());
        }
    }

    struct Chunks {
        pieces: std::collections::VecDeque<io::Result<Vec<u8>>>,
        reads: usize,
    }
    impl Read for Chunks {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.reads += 1;
            let bytes = self
                .pieces
                .pop_front()
                .unwrap_or_else(|| Err(io::ErrorKind::WouldBlock.into()))?;
            buffer[..bytes.len()].copy_from_slice(&bytes);
            Ok(bytes.len())
        }
    }

    #[test]
    fn independent_header_readers_preserve_progress_and_absolute_deadlines() {
        let start = Instant::now();
        let mut waiting = HeaderReader::new(start);
        let mut blocked = Chunks {
            pieces: Default::default(),
            reads: 0,
        };
        assert!(waiting.poll(&mut blocked, || start).unwrap().is_none());
        let mut ready = HeaderReader::new(start);
        let request = b"GET /api/model?name=caf%C3%A9 HTTP/1.1\r\nHost: localhost:8797\r\n\r\n";
        let raw = ready
            .poll(&mut io::Cursor::new(request), || start)
            .unwrap()
            .unwrap();
        let (method, path, query, _) = parse_request(&raw, 8797).unwrap();
        assert_eq!(method, "GET");
        assert_eq!(path, "/api/model");
        assert_eq!(query.get("name", ""), "café");
        assert!(waiting
            .poll(&mut blocked, || start + HEADER_DEADLINE)
            .is_err());
        assert_eq!(blocked.reads, 1); // Expiration does not perform another read.
    }

    #[test]
    fn header_progress_never_renews_deadline_or_exceeds_read_budget() {
        let start = Instant::now();
        let mut reader = HeaderReader::new(start);
        let mut chunks = Chunks {
            pieces: (0..9).map(|_| Ok(vec![b'a'; 1024])).collect(),
            reads: 0,
        };
        assert!(reader.poll(&mut chunks, || start).unwrap().is_none());
        assert_eq!(chunks.reads, 4);
        assert!(reader.poll(&mut chunks, || start).unwrap().is_none());
        assert_eq!(chunks.reads, 8);
        assert!(reader.poll(&mut chunks, || start).is_err());
        assert_eq!(reader.raw.len(), HEADER_LIMIT);
        let mut reader = HeaderReader::new(start);
        let mut chunks = Chunks {
            pieces: [Ok(b"GET / HTTP/1.1\r\n".to_vec())].into(),
            reads: 0,
        };
        assert!(reader
            .poll(&mut chunks, || start + Duration::from_millis(400))
            .unwrap()
            .is_none());
        assert!(reader
            .poll(&mut chunks, || start + Duration::from_millis(501))
            .is_err());
    }

    #[test]
    fn accept_error_policy_retries_recoverable_errors_with_capped_backoff() {
        for kind in [
            io::ErrorKind::Interrupted,
            io::ErrorKind::ConnectionAborted,
            io::ErrorKind::ConnectionReset,
        ] {
            assert!(recoverable_accept(&kind.into()));
        }
        for code in [libc::EMFILE, libc::ENFILE, libc::ENOBUFS, libc::ENOMEM] {
            assert!(recoverable_accept(&io::Error::from_raw_os_error(code)));
        }
        assert!(!recoverable_accept(&io::ErrorKind::PermissionDenied.into()));
        for failures in 0..100 {
            assert!(accept_backoff(failures) >= Duration::from_millis(10));
            assert!(accept_backoff(failures) <= Duration::from_millis(250));
        }
    }

    #[test]
    fn one_response_deadline_covers_partial_header_and_body_writes() {
        use std::cell::{Cell, RefCell};
        struct Writer<'a> {
            clock: &'a Cell<Instant>,
            budgets: RefCell<Vec<Duration>>,
            bytes: Vec<u8>,
        }
        impl Write for Writer<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.push(bytes[0]);
                self.clock
                    .set(self.clock.get() + Duration::from_millis(800));
                Ok(1)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        impl TimedWrite for Writer<'_> {
            fn write_timeout(&self, timeout: Duration) -> io::Result<()> {
                self.budgets.borrow_mut().push(timeout);
                Ok(())
            }
        }
        let start = Instant::now();
        let clock = Cell::new(start);
        let mut writer = Writer {
            clock: &clock,
            budgets: RefCell::new(Vec::new()),
            bytes: Vec::new(),
        };
        let error = write_response(
            &mut writer,
            &[b"hh", b"body"],
            start + RESPONSE_DEADLINE,
            || clock.get(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert_eq!(writer.bytes, b"hhbo");
        assert_eq!(
            *writer.budgets.borrow(),
            vec![
                Duration::from_millis(3000),
                Duration::from_millis(2200),
                Duration::from_millis(1400),
                Duration::from_millis(600)
            ]
        );
    }

    #[test]
    fn blocked_response_does_not_consume_another_readers_deadline() {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            mpsc::{sync_channel, Receiver, SyncSender},
        };
        struct Writer {
            elapsed: Arc<AtomicU64>,
            entered: Option<SyncSender<()>>,
            release: Receiver<()>,
            bytes: Vec<u8>,
        }
        impl Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if let Some(entered) = self.entered.take() {
                    entered.send(()).unwrap();
                    self.release.recv_timeout(Duration::from_secs(2)).unwrap();
                    self.elapsed.store(600, Ordering::SeqCst);
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        impl TimedWrite for Writer {
            fn write_timeout(&self, _: Duration) -> io::Result<()> {
                Ok(())
            }
        }
        let start = Instant::now();
        let elapsed = Arc::new(AtomicU64::new(0));
        let (entered, waiting) = sync_channel(0);
        let (release, barrier) = sync_channel(0);
        let worker_clock = elapsed.clone();
        let writer = std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(move || {
                let mut writer = Writer {
                    elapsed: worker_clock.clone(),
                    entered: Some(entered),
                    release: barrier,
                    bytes: Vec::new(),
                };
                write_response(
                    &mut writer,
                    &[b"headers", b"body"],
                    start + RESPONSE_DEADLINE,
                    || start + Duration::from_millis(worker_clock.load(Ordering::SeqCst)),
                )
                .unwrap();
                writer.bytes
            })
            .unwrap();
        waiting.recv_timeout(Duration::from_secs(2)).unwrap();
        elapsed.store(100, Ordering::SeqCst);
        let request = b"GET /api/model HTTP/1.1\r\nHost: localhost:8797\r\n\r\n";
        let mut pending = vec![PendingHeader {
            socket: io::Cursor::new(request.to_vec()),
            reader: HeaderReader::new(start),
        }];
        let (send, receive) = sync_channel(DISPATCH_CAPACITY);
        for (_, result) in collect_headers(&mut pending, || {
            start + Duration::from_millis(elapsed.load(Ordering::SeqCst))
        }) {
            send.try_send(result.unwrap()).unwrap();
        }
        assert!(pending.is_empty());
        release.send(()).unwrap();
        assert_eq!(writer.join().unwrap(), b"headersbody");
        assert_eq!(elapsed.load(Ordering::SeqCst), 600);
        // The second request was complete/queued before the slow response ended.
        let raw = receive.try_recv().unwrap();
        assert_eq!(parse_request(&raw, 8797).unwrap().1, "/api/model");
    }
}

#[cfg(test)]
mod dispatch_vectors {
    //! Tiny direct-dispatch vectors: no listener loop, numeric worker or inference.
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Fixture {
        state: Arc<State>,
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "atlas-dispatch-vectors-{}-{}",
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
            let state = Arc::new(
                State::open(
                    &model,
                    &root.join("cache"),
                    Some("dispatch fixture".into()),
                    Some("fixture-v1".into()),
                )
                .unwrap(),
            );
            Self { state, root }
        }

        fn request(&self, path: &str) -> (String, Value) {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let address = listener.local_addr().unwrap();
            let mut client = TcpStream::connect_timeout(&address, Duration::from_secs(3)).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let (socket, _) = listener.accept().unwrap();
            let (sender, _receiver) = sync_channel(8);
            let raw = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\n\r\n");
            dispatch(
                &self.state,
                &sender,
                Connection {
                    socket,
                    lease: None,
                    eligible: false,
                },
                raw.as_bytes(),
                address.port(),
            );
            let mut bytes = Vec::new();
            client.read_to_end(&mut bytes).unwrap();
            assert!(bytes.len() < 64 * 1024, "tiny dispatch response bound");
            let response = String::from_utf8(bytes).unwrap();
            let (head, body) = response.split_once("\r\n\r\n").unwrap();
            let status = head.lines().next().unwrap().to_owned();
            (status, serde_json::from_str(body).unwrap())
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn model_route_keeps_catalog_status_and_fields() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (status, body) = fixture.request("/api/model");
        assert_eq!(status, "HTTP/1.1 200 OK", "viewer model route status");
        assert_eq!(body["api_version"], 1);
        assert_eq!(body["name"], "dispatch fixture");
        assert_eq!(body["revision"], "fixture-v1");
        assert_eq!(body["parameter_count"], 2);
        assert_eq!(body["catalog"][0]["name"], "weights");
        assert_eq!(body["catalog"][0]["dtype"], "BF16");
        assert_eq!(body["calibration_complete"], false);
    }

    #[test]
    fn inspect_route_keeps_selected_native_cell_and_raw_bytes() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (status, body) = fixture.request("/api/inspect?tensor=0&row=0&col=1");
        assert_eq!(status, "HTTP/1.1 200 OK", "viewer inspect route status");
        assert_eq!(body["row"], 0);
        assert_eq!(body["col"], 1);
        assert_eq!(body["native_indices"], json!([0, 1]));
        assert_eq!(body["raw_exact"], "-1");
        assert_eq!(body["raw_hex_le"], "80bf");
        assert_eq!(body["classification"], "finite");
        assert_eq!(body["transforms_ready"], false);
    }

    #[test]
    fn unknown_asset_keeps_exact_not_found_response() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        let (status, body) = fixture.request("/missing-fixture-asset");
        assert_eq!(
            status, "HTTP/1.1 404 Not Found",
            "viewer missing asset status"
        );
        assert_eq!(
            body,
            json!({"api_version":1,"error":"Not found"}),
            "viewer missing asset body"
        );
    }

    impl Fixture {
        // One owned socket pair, no listener loop or numeric worker. The closed
        // receiver holds queue refusals without leaving an enqueued socket open.
        fn declaration_request(&self, method: &str, path: &str, header: &str) -> (String, Value) {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let address = listener.local_addr().unwrap();
            let mut client = TcpStream::connect_timeout(&address, Duration::from_secs(3)).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let (socket, _) = listener.accept().unwrap();
            let (sender, receiver) = sync_channel(1);
            drop(receiver);
            let raw = format!("{method} {path} HTTP/1.1\r\nHost: {address}\r\n{header}\r\n");
            dispatch(
                &self.state,
                &sender,
                Connection {
                    socket,
                    lease: None,
                    eligible: false,
                },
                raw.as_bytes(),
                address.port(),
            );
            let mut bytes = Vec::new();
            client.read_to_end(&mut bytes).unwrap();
            assert!(bytes.len() < 64 * 1024, "bounded declaration response");
            let response = String::from_utf8(bytes).unwrap();
            let (head, body) = response.split_once("\r\n\r\n").unwrap();
            (
                head.lines().next().unwrap().into(),
                serde_json::from_str(body).unwrap(),
            )
        }
    }

    #[test]
    fn declaration_routes_keep_status_method_and_local_action_refusals() {
        let _isolation = crate::resources::test_workspace_guard();
        let fixture = Fixture::new();
        for (method, path, header, status, message) in [
            (
                "GET",
                "/api/tensor-status",
                "",
                "HTTP/1.1 400 Bad Request",
                "Selected tensor required",
            ),
            (
                "GET",
                "/api/progress?tensor=bad",
                "",
                "HTTP/1.1 400 Bad Request",
                "invalid digit found in string",
            ),
            (
                "POST",
                "/api/model",
                "",
                "HTTP/1.1 400 Bad Request",
                "Only GET and local calibration POST are supported",
            ),
            (
                "POST",
                "/api/calibrate?tensor=0",
                "",
                "HTTP/1.1 400 Bad Request",
                "Local action header required",
            ),
            (
                "POST",
                "/api/calibrate?tensor=0",
                "X-Atlas-Local: 1\r\n",
                "HTTP/1.1 503 Service Unavailable",
                "Numeric queue full; retry shortly",
            ),
            (
                "GET",
                "/tile?tensor=0",
                "",
                "HTTP/1.1 503 Service Unavailable",
                "Numeric queue full; retry shortly",
            ),
            (
                "GET",
                "/api/view?tensor=bad",
                "",
                "HTTP/1.1 400 Bad Request",
                "invalid digit found in string",
            ),
        ] {
            let (observed, body) = fixture.declaration_request(method, path, header);
            assert_eq!(observed, status, "viewer declaration status: {path}");
            assert_eq!(
                body,
                json!({"api_version":1,"error":message}),
                "viewer declaration refusal: {path}"
            );
        }
        for path in [
            "/api/progress",
            "/api/progress?tensor=0",
            "/api/tensor-status?tensor=0",
        ] {
            let (status, body) = fixture.request(path);
            assert_eq!(
                status, "HTTP/1.1 200 OK",
                "viewer declaration status success: {path}"
            );
            assert_eq!(body["api_version"], 1);
        }
        let (status, body) = fixture.request("/api/model?retained-extra=value");
        assert_eq!(
            status, "HTTP/1.1 200 OK",
            "viewer existing extra query behavior"
        );
        assert_eq!(body["name"], "dispatch fixture");
    }
}

#[cfg(test)]
#[path = "../tests/support/native_error_boundary.rs"]
mod error_boundary_vectors;

#[cfg(test)]
#[path = "../tests/support/native_view_response.rs"]
mod view_response_vectors;

#[cfg(test)]
#[path = "../tests/support/native_calibration_admission.rs"]
mod calibration_admission_vectors;
