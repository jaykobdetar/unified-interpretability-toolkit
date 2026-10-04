//! Standalone-only owned sockets and bounded OS-woken return queue.
use super::{reuse, HeaderReader, PendingHeader, MAX_PENDING_HEADERS};
use crate::{require, Result};
use std::{
    collections::VecDeque,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    os::fd::{AsRawFd, RawFd},
    os::unix::net::UnixStream,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
        Arc, Mutex,
    },
    time::Instant,
};

pub(super) struct Connection {
    pub(super) socket: TcpStream,
    pub(super) lease: Option<reuse::Lease>,
    pub(super) eligible: bool,
}
impl Read for Connection {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.socket.read(bytes)
    }
}
impl Connection {
    fn fresh(socket: TcpStream) -> Self {
        Self {
            socket,
            lease: None,
            eligible: false,
        }
    }
    pub(super) fn expired(&self, now: Instant) -> bool {
        self.lease.as_ref().is_some_and(|l| !l.admission_open(now))
    }
    pub(super) fn close_error(self, status: u16, error: impl std::fmt::Display) {
        let Self {
            socket,
            lease: _lease,
            ..
        } = self;
        super::error(socket, status, error);
    }
    fn reject(self, status: u16, code: &str, message: &str) {
        let Self {
            socket,
            lease: _lease,
            ..
        } = self;
        super::reject_now(socket, status, code, message);
    }
    fn close_png(self, body: &[u8], headers: &str) {
        let Self {
            socket,
            lease: _lease,
            ..
        } = self;
        super::reply(socket, 200, "image/png", body, headers);
    }
}
trait QueueItem {
    fn id(&self) -> Option<u64>;
}
impl QueueItem for Connection {
    fn id(&self) -> Option<u64> {
        self.lease.as_ref().map(|l| l.id)
    }
}
struct BoundedQueue<T>(Mutex<VecDeque<T>>);
impl<T: QueueItem> BoundedQueue<T> {
    fn new() -> Self {
        Self(Mutex::new(VecDeque::with_capacity(reuse::MAX_CONNECTIONS)))
    }
    fn push(&self, value: T) -> Option<u64> {
        let id = value.id()?;
        let mut queue = self.0.lock().ok()?;
        if queue.len() >= reuse::MAX_CONNECTIONS || queue.iter().any(|v| v.id() == Some(id)) {
            return None;
        }
        queue.push_back(value);
        Some(id)
    }
    fn pop(&self) -> Result<Option<T>> {
        Ok(self
            .0
            .lock()
            .map_err(|_| "Reuse queue unavailable")?
            .pop_front())
    }
    fn remove(&self, id: u64) {
        let removed = {
            let mut queue = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let position = queue.iter().position(|v| v.id() == Some(id));
            position.and_then(|i| queue.remove(i))
        };
        drop(removed);
    }
}
static RUNTIME_ACTIVE: AtomicBool = AtomicBool::new(false);
struct RuntimeOwner;
impl Drop for RuntimeOwner {
    fn drop(&mut self) {
        RUNTIME_ACTIVE.store(false, Ordering::Release);
    }
}
pub(super) struct Runtime {
    queue: BoundedQueue<Connection>,
    wake_read: UnixStream,
    wake_write: UnixStream,
    // Last field: owned queue/socket fields close before releasing singleton admission.
    _owner: RuntimeOwner,
}
// The observed value includes Linux's doubling; all syscall storage is initialized
// and the descriptor stays owned across set/readback. No unchecked raw ownership.
fn locked_buffer(
    fd: RawFd,
    option: libc::c_int,
    request: libc::c_int,
    maximum: u64,
) -> Result<u64> {
    let mut observed: libc::c_int = 0;
    let mut length = std::mem::size_of_val(&observed) as libc::socklen_t;
    require(
        unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                option,
                (&request as *const libc::c_int).cast(),
                std::mem::size_of_val(&request) as libc::socklen_t,
            )
        } == 0,
        "Cannot set bounded reuse socket buffer",
    )?;
    require(
        unsafe {
            libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                option,
                (&mut observed as *mut libc::c_int).cast(),
                &mut length,
            )
        } == 0,
        "Cannot measure reuse socket buffer",
    )?;
    require(
        length as usize == std::mem::size_of_val(&observed)
            && observed > 0
            && observed as u64 <= maximum,
        "Measured reuse socket buffer exceeds budget",
    )?;
    Ok(observed as u64)
}
impl Runtime {
    pub(super) fn new() -> Result<Arc<Self>> {
        require(
            crate::resources::standalone_active(),
            "Reuse applies only to standalone mode",
        )?;
        require(
            RUNTIME_ACTIVE
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "One process-global reuse transport is already active",
        )?;
        let owner = RuntimeOwner;
        let (read, write) = UnixStream::pair()?;
        for stream in [&read, &write] {
            stream.set_nonblocking(true)?;
            locked_buffer(stream.as_raw_fd(), libc::SO_SNDBUF, 4096, 8192)?;
            locked_buffer(stream.as_raw_fd(), libc::SO_RCVBUF, 4096, 8192)?;
        }
        Ok(Arc::new(Self {
            queue: BoundedQueue::new(),
            wake_read: read,
            wake_write: write,
            _owner: owner,
        }))
    }
    fn pop(&self) -> Result<Option<Connection>> {
        self.queue.pop()
    }
    fn drain_wake(&self) -> Result<()> {
        for _ in 0..4 {
            let mut bytes = [0u8; 16];
            match (&self.wake_read).read(&mut bytes) {
                Ok(0) => return Err("Reuse notification closed".into()),
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
    fn offer(&self, connection: Connection) {
        let Some(deadline) = connection.lease.as_ref().and_then(|l| l.header_deadline()) else {
            return;
        };
        let Some(id) = self.queue.push(connection) else {
            return;
        };
        let mut writer = &self.wake_write;
        if signal(&mut writer, deadline, Instant::now) {
            return;
        }
        // Cancel only our unique owned item. Intake may already own it; no FD map.
        self.queue.remove(id);
    }
    fn admit(&self, socket: &TcpStream, now: Instant) -> Option<reuse::Lease> {
        let send = locked_buffer(
            socket.as_raw_fd(),
            libc::SO_SNDBUF,
            64 * 1024,
            reuse::SEND_BUFFER_MAX,
        )
        .ok()?;
        let receive = locked_buffer(
            socket.as_raw_fd(),
            libc::SO_RCVBUF,
            32 * 1024,
            reuse::RECEIVE_BUFFER_MAX,
        )
        .ok()?;
        let available = crate::headroom().ok()?;
        reuse::Ledger::global().admit(
            now,
            send,
            receive,
            available,
            crate::resources::policy().available_floor_bytes,
        )
    }
}
// All writes are one byte, nonblocking at runtime, and have finite retry work.
fn signal(writer: &mut impl Write, deadline: Instant, mut now: impl FnMut() -> Instant) -> bool {
    for _ in 0..4 {
        if now() >= deadline {
            return false;
        }
        match writer.write(&[1]) {
            Ok(1) => return true,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return true,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            _ => return false,
        }
    }
    false
}
fn can_admit_returned(pending: usize, reused: usize) -> bool {
    pending < MAX_PENDING_HEADERS && reused < MAX_PENDING_HEADERS - 1
}

fn no_waiting_input(socket: &TcpStream) -> bool {
    if socket.set_nonblocking(true).is_err() {
        return false;
    }
    let result = socket.peek(&mut [0u8; 1]);
    // Unknown/queued input disables reuse; ordinary close responses remain available.
    matches!(result, Err(e) if e.kind() == io::ErrorKind::WouldBlock)
}
pub(super) fn reply_tile(
    mut connection: Connection,
    body: &[u8],
    headers: &str,
    runtime: Option<&Runtime>,
) {
    let Some(runtime) = runtime else {
        connection.close_png(body, headers);
        return;
    };
    if !connection.eligible || body.len() > 1024 * 1024 || !no_waiting_input(&connection.socket) {
        connection.close_png(body, headers);
        return;
    }
    if connection.lease.is_none() {
        connection.lease = runtime.admit(&connection.socket, Instant::now());
    }
    let Some(lease) = connection.lease.as_ref() else {
        connection.close_png(body, headers);
        return;
    };
    let now = Instant::now();
    if !lease.reusable_response(now) {
        connection.close_png(body, headers);
        return;
    }
    let Some(deadline) = lease.write_deadline(now, true) else {
        return;
    };
    let head = super::response_head_with_connection(200, "image/png", body.len(), headers, true);
    if connection.socket.set_nonblocking(false).is_err()
        || super::write_response(
            &mut connection.socket,
            &[head.as_bytes(), body],
            deadline,
            Instant::now,
        )
        .is_err()
    {
        return;
    }
    if connection
        .lease
        .as_mut()
        .is_some_and(|l| l.completed_response(Instant::now()))
    {
        runtime.offer(connection);
    }
}

pub(super) type Completed = (Connection, Vec<u8>);
pub(super) fn run(
    listener: TcpListener,
    sender: SyncSender<Completed>,
    runtime: Option<Arc<Runtime>>,
) -> Result<()> {
    listener.set_nonblocking(true)?;
    let mut pending: Vec<PendingHeader<Connection>> = Vec::with_capacity(MAX_PENDING_HEADERS);
    let mut next_accept = Instant::now();
    let mut failures = 0u32;
    let mut prefer_returned = true;
    loop {
        for (connection, result) in super::collect_headers(&mut pending, Instant::now) {
            match result {
                Ok(raw) => {
                    if let Err(e) = sender.try_send((connection, raw)) {
                        let (std::sync::mpsc::TrySendError::Full((connection, _))
                        | std::sync::mpsc::TrySendError::Disconnected((connection, _))) = e;
                        connection.reject(
                            503,
                            "dispatch_full",
                            "Request dispatch full; retry shortly",
                        );
                    }
                }
                Err(e) => connection.reject(400, "request_input", &e.to_string()),
            }
        }
        if let Some(runtime) = &runtime {
            runtime.drain_wake()?;
        }
        // Constant bounded admission work, alternating choices even under readiness.
        for _ in 0..MAX_PENDING_HEADERS * 2 {
            prefer_returned = !prefer_returned;
            if prefer_returned {
                if let Some(runtime) = &runtime {
                    if let Some(connection) = runtime.pop()? {
                        let now = Instant::now();
                        let deadline = connection.lease.as_ref().and_then(|l| l.header_deadline());
                        let reused = pending.iter().filter(|p| p.socket.lease.is_some()).count();
                        if can_admit_returned(pending.len(), reused)
                            && !connection.expired(now)
                            && deadline.is_some_and(|d| now < d)
                            && connection.socket.set_nonblocking(true).is_ok()
                        {
                            let mut reader = HeaderReader::new(now);
                            reader.deadline = deadline.unwrap();
                            pending.push(PendingHeader {
                                socket: connection,
                                reader,
                            });
                        } // Otherwise close/drop the still-owned returned socket and charge.
                    }
                }
            } else if Instant::now() >= next_accept {
                match listener.accept() {
                    Ok((socket, _)) => {
                        failures = 0;
                        if socket.set_nonblocking(true).is_err() {
                            continue;
                        }
                        let connection = Connection::fresh(socket);
                        if pending.len() >= MAX_PENDING_HEADERS {
                            connection.reject(
                                503,
                                "admission_full",
                                "Request admission full; retry shortly",
                            );
                        } else {
                            pending.push(PendingHeader {
                                socket: connection,
                                reader: HeaderReader::new(Instant::now()),
                            });
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(e) if super::recoverable_accept(&e) => {
                        failures = failures.saturating_add(1).min(6);
                        next_accept = Instant::now() + super::accept_backoff(failures);
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }
        wait(&listener, &pending, next_accept, runtime.as_deref())?;
    }
}
fn wait(
    listener: &TcpListener,
    pending: &[PendingHeader<Connection>],
    retry: Instant,
    runtime: Option<&Runtime>,
) -> Result<()> {
    let now = Instant::now();
    let retrying = retry > now;
    let mut descriptors = Vec::with_capacity(MAX_PENDING_HEADERS + 2);
    if !retrying {
        descriptors.push(libc::pollfd {
            fd: listener.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });
    }
    if let Some(runtime) = runtime {
        descriptors.push(libc::pollfd {
            fd: runtime.wake_read.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });
    }
    descriptors.extend(pending.iter().map(|p| libc::pollfd {
        fd: p.socket.socket.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    }));
    let timeout = super::poll_timeout(
        now,
        pending
            .iter()
            .map(|p| p.reader.deadline)
            .chain(retrying.then_some(retry)),
    );
    // SAFETY: descriptors remain owned and the initialized vector is writable.
    let result = unsafe {
        libc::poll(
            descriptors.as_mut_ptr(),
            descriptors.len() as libc::nfds_t,
            timeout,
        )
    };
    if result < 0 {
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e.into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    struct Item {
        id: u64,
        dropped: Arc<AtomicUsize>,
    }
    impl QueueItem for Item {
        fn id(&self) -> Option<u64> {
            Some(self.id)
        }
    }
    impl Drop for Item {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[test]
    fn full_and_cancelled_return_items_close_only_their_owned_envelopes() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let queue = BoundedQueue::new();
        for id in 1..=4 {
            assert_eq!(
                queue.push(Item {
                    id,
                    dropped: dropped.clone()
                }),
                Some(id)
            );
        }
        assert!(queue
            .push(Item {
                id: 5,
                dropped: dropped.clone()
            })
            .is_none());
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        queue.remove(3);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
        queue.remove(3);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
        assert_eq!(queue.pop().unwrap().unwrap().id, 1);
        assert_eq!(dropped.load(Ordering::Relaxed), 3);
        drop(queue);
        assert_eq!(dropped.load(Ordering::Relaxed), 5);
    }
    #[test]
    fn returned_admission_keeps_one_slot_for_fresh_progress() {
        for pending in 0..=4 {
            for reused in 0..=pending {
                if can_admit_returned(pending, reused) {
                    let resulting_header_count = pending + 1;
                    assert!(resulting_header_count <= MAX_PENDING_HEADERS);
                    assert!(reused + 1 < MAX_PENDING_HEADERS);
                }
            }
        }
        assert!(!can_admit_returned(4, 0));
        assert!(!can_admit_returned(3, 3));
        assert!(can_admit_returned(3, 2));
    }
    struct Interrupted {
        writes: usize,
    }
    impl Write for Interrupted {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            assert_eq!(bytes, &[1]);
            self.writes += 1;
            Err(io::ErrorKind::Interrupted.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn notification_retry_and_expiration_are_bounded_without_socket_traffic() {
        let now = Instant::now();
        let mut writer = Interrupted { writes: 0 };
        assert!(!signal(&mut writer, now, || now));
        assert_eq!(writer.writes, 0);
        assert!(!signal(&mut writer, now + reuse::HEADER_WINDOW, || now));
        assert_eq!(writer.writes, 4);
    }
}
