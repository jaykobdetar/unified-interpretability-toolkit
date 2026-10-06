//! Bounded standalone tile reuse policy. Socket ownership remains exclusive.
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub(super) const MAX_CONNECTIONS: usize = 4;
pub(super) const MAX_REQUESTS: u8 = 8;
pub(super) const HEADER_WINDOW: Duration = Duration::from_millis(500);
pub(super) const WRITE_WINDOW: Duration = Duration::from_secs(3);
pub(super) const ELIGIBILITY_WINDOW: Duration = Duration::from_secs(10);
const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const TOTAL_BYTES: u64 = 8 * MIB;
const LEASE_BYTES_MAX: u64 = 2 * MIB;
const FIXED_BYTES: u64 = 64 * KIB;
const REQUEST_BYTES: u64 = 512 * KIB;
const RESPONSE_BYTES: u64 = MIB;
const METADATA_BYTES: u64 = 16 * KIB;
pub(super) const SEND_BUFFER_MAX: u64 = 128 * KIB;
pub(super) const RECEIVE_BUFFER_MAX: u64 = 64 * KIB;

#[derive(Default)]
struct Counts {
    active: usize,
    bytes: u64,
    next_id: u64,
}
pub(super) struct Ledger(Mutex<Counts>);
impl Ledger {
    fn new() -> Self {
        Self(Mutex::new(Counts {
            bytes: FIXED_BYTES,
            ..Default::default()
        }))
    }
    pub(super) fn global() -> Arc<Self> {
        static LEDGER: OnceLock<Arc<Ledger>> = OnceLock::new();
        LEDGER.get_or_init(|| Arc::new(Self::new())).clone()
    }
    pub(super) fn admit(
        self: &Arc<Self>,
        now: Instant,
        send: u64,
        receive: u64,
        available: u64,
        floor: u64,
    ) -> Option<Lease> {
        // Observed kernel sizes include socket-buffer doubling. Autotuning must
        // already be locked to the requested finite buffers before this call.
        if send == 0 || send > SEND_BUFFER_MAX || receive == 0 || receive > RECEIVE_BUFFER_MAX {
            return None;
        }
        let charge = REQUEST_BYTES
            .checked_add(RESPONSE_BYTES)?
            .checked_add(METADATA_BYTES)?
            .checked_add(send)?
            .checked_add(receive)?;
        let expires = now.checked_add(ELIGIBILITY_WINDOW)?;
        let mut counts = self.0.lock().ok()?;
        let next = counts.bytes.checked_add(charge)?;
        // Reserve every live lease's conservative future footprint above the
        // measured effective free-RAM floor, including queued/idle ownership.
        let required = floor.checked_add(next)?;
        if available < required {
            return None;
        }
        if counts.active >= MAX_CONNECTIONS || charge > LEASE_BYTES_MAX || next > TOTAL_BYTES {
            return None;
        }
        let id = counts.next_id.checked_add(1)?;
        counts.active += 1;
        counts.bytes = next;
        counts.next_id = id;
        Some(Lease {
            ledger: self.clone(),
            charge,
            id,
            expires,
            served: 0,
            next_header: None,
        })
    }
}
// Not Clone: exactly one owned connection carries the charge through every stage.
pub(super) struct Lease {
    ledger: Arc<Ledger>,
    charge: u64,
    pub(super) id: u64,
    expires: Instant,
    served: u8,
    next_header: Option<Instant>,
}
impl Lease {
    pub(super) fn admission_open(&self, now: Instant) -> bool {
        now < self.expires && self.served < MAX_REQUESTS
    }
    pub(super) fn reusable_response(&self, now: Instant) -> bool {
        self.admission_open(now) && self.served + 1 < MAX_REQUESTS
    }
    pub(super) fn write_deadline(&self, now: Instant, reusable: bool) -> Option<Instant> {
        let deadline = now.checked_add(WRITE_WINDOW)?;
        // Expired active numeric work may finish and send an ordinary close
        // response under the existing write window. This is no hard IO timer.
        Some(if reusable {
            deadline.min(self.expires)
        } else {
            deadline
        })
    }
    pub(super) fn completed_response(&mut self, now: Instant) -> bool {
        if self.served >= MAX_REQUESTS {
            return false;
        }
        self.served += 1;
        if !self.admission_open(now) {
            return false;
        }
        self.next_header = now.checked_add(HEADER_WINDOW).map(|d| d.min(self.expires));
        self.next_header.is_some_and(|d| now < d)
    }
    pub(super) fn header_deadline(&self) -> Option<Instant> {
        self.next_header
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        // Recovery only releases an already-owned reservation after a panic.
        let mut counts = self.ledger.0.lock().unwrap_or_else(|e| e.into_inner());
        counts.active -= 1;
        counts.bytes -= self.charge;
    }
}

/// Persistent negotiation is deliberately narrower than ordinary parsing.
/// Exactly one complete bodyless HTTP/1.1 header is required; tail bytes close.
pub(super) fn bounded_framing(raw: &[u8]) -> bool {
    if raw.len() > super::HEADER_LIMIT {
        return false;
    }
    let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let Ok(text) = std::str::from_utf8(raw) else {
        return false;
    };
    let mut lines = text.split("\r\n");
    let Some(first) = lines.next() else {
        return false;
    };
    let mut parts = first.split_whitespace();
    let method = parts.next();
    let url = parts.next();
    let version = parts.next();
    end + 4 == raw.len()
        && method.is_some()
        && version.is_some()
        && parts.next().is_none()
        && lines.take_while(|line| !line.is_empty()).count() <= 32
        && url.is_some_and(|url| url.split_once('?').map_or(0, |(_, q)| q.split('&').count()) <= 16)
}
pub(super) fn framing_candidate(raw: &[u8]) -> bool {
    bounded_framing(raw)
        && std::str::from_utf8(raw)
            .ok()
            .and_then(|text| text.split("\r\n").next())
            .is_some_and(|first| {
                first.split_whitespace().next() == Some("GET")
                    && first.split_whitespace().nth(2) == Some("HTTP/1.1")
            })
}
pub(super) fn candidate(
    raw: &[u8],
    method: &str,
    path: &str,
    binding: &str,
    connection: &str,
) -> bool {
    framing_candidate(raw)
        && method == "GET"
        && path == crate::api::viewer::TILE
        && binding.len() == 64
        && binding
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && !connection
            .split(',')
            .any(|s| s.trim().eq_ignore_ascii_case("close"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_stages_keep_the_finite_charge_and_drop_releases_it() {
        let ledger = Arc::new(Ledger::new());
        let now = Instant::now();
        let mut leases = (0..MAX_CONNECTIONS)
            .map(|_| {
                ledger
                    .admit(now, SEND_BUFFER_MAX, RECEIVE_BUFFER_MAX, u64::MAX, 0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(ledger
            .admit(now, SEND_BUFFER_MAX, RECEIVE_BUFFER_MAX, u64::MAX, 0)
            .is_none());
        let counts = ledger.0.lock().unwrap();
        assert_eq!(counts.active, 4);
        assert!(counts.bytes <= TOTAL_BYTES);
        drop(counts);
        let mut moved = leases.pop().unwrap();
        let before = ledger.0.lock().unwrap().bytes;
        assert!(moved.completed_response(now));
        assert_eq!(ledger.0.lock().unwrap().bytes, before);
        drop(moved);
        assert!(ledger
            .admit(now, SEND_BUFFER_MAX, RECEIVE_BUFFER_MAX, u64::MAX, 0)
            .is_some());
        drop(leases);
        assert_eq!(ledger.0.lock().unwrap().active, 0);
        assert_eq!(ledger.0.lock().unwrap().bytes, FIXED_BYTES);
    }
    #[test]
    fn measured_buffers_and_count_overflow_refuse_without_accounting_changes() {
        let ledger = Arc::new(Ledger::new());
        for (send, receive) in [
            (0, 1),
            (1, 0),
            (SEND_BUFFER_MAX + 1, 1),
            (1, RECEIVE_BUFFER_MAX + 1),
            (u64::MAX, 1),
        ] {
            assert!(ledger
                .admit(Instant::now(), send, receive, u64::MAX, 0)
                .is_none());
            assert_eq!(ledger.0.lock().unwrap().bytes, FIXED_BYTES);
        }
        ledger.0.lock().unwrap().next_id = u64::MAX;
        assert!(ledger.admit(Instant::now(), 1, 1, u64::MAX, 0).is_none());
        assert_eq!(ledger.0.lock().unwrap().active, 0);
    }
    #[test]
    fn request_cap_and_absolute_horizon_do_not_extend_on_progress() {
        let ledger = Arc::new(Ledger::new());
        let start = Instant::now();
        let mut lease = ledger.admit(start, 1, 1, u64::MAX, 0).unwrap();
        for served in 0u8..8 {
            let now = start + Duration::from_millis(u64::from(served) * 20);
            assert_eq!(lease.reusable_response(now), served < 7);
            assert_eq!(lease.completed_response(now), served < 7);
        }
        assert!(!lease.admission_open(start));
        let mut lease = ledger.admit(start, 1, 1, u64::MAX, 0).unwrap();
        let near = start + ELIGIBILITY_WINDOW - Duration::from_millis(1);
        assert!(lease.completed_response(near));
        assert_eq!(lease.header_deadline(), Some(start + ELIGIBILITY_WINDOW));
        assert!(!lease.admission_open(start + ELIGIBILITY_WINDOW));
        // Existing active numerical IO gets a close response, not a fake timer.
        let after = start + ELIGIBILITY_WINDOW + Duration::from_secs(2);
        assert!(!lease.reusable_response(after));
        assert_eq!(
            lease.write_deadline(after, false),
            Some(after + WRITE_WINDOW)
        );
    }
    #[test]
    fn only_single_bound_http11_tile_headers_negotiate_reuse() {
        let binding = "a".repeat(64);
        let raw = b"GET /tile HTTP/1.1\r\nHost: localhost:8775\r\n\r\n";
        assert!(candidate(raw, "GET", "/tile", &binding, "keep-alive"));
        assert!(!candidate(
            raw,
            "GET",
            "/tile",
            &binding,
            "Keep-Alive, CLOSE"
        ));
        assert!(!candidate(raw, "GET", "/api/view", &binding, ""));
        assert!(!candidate(raw, "POST", "/tile", &binding, ""));
        assert!(!candidate(raw, "GET", "/tile", "stale", ""));
        let mut tail = raw.to_vec();
        tail.extend_from_slice(raw);
        assert!(!candidate(&tail, "GET", "/tile", &binding, ""));
        assert!(!candidate(
            b"GET /tile HTTP/1.0\r\n\r\n",
            "GET",
            "/tile",
            &binding,
            ""
        ));
    }
    #[test]
    fn bounded_api_transition_is_allowed_to_close_without_reuse() {
        let post = b"POST /api/calibrate HTTP/1.1\r\nHost: localhost:8775\r\nX-Atlas-Local: 1\r\nContent-Length: 0\r\n\r\n";
        assert!(bounded_framing(post));
        assert!(!candidate(
            post,
            "POST",
            "/api/calibrate",
            &"a".repeat(64),
            ""
        ));
        assert!(!framing_candidate(post));
        let many = format!(
            "GET /tile?{} HTTP/1.1\r\n\r\n",
            (0..17)
                .map(|i| format!("x{i}=0"))
                .collect::<Vec<_>>()
                .join("&")
        );
        assert!(!bounded_framing(many.as_bytes()));
        let tail = b"GET /tile HTTP/1.1 extra\r\n\r\n";
        assert!(!bounded_framing(tail));
    }
    #[test]
    fn effective_ram_admission_reserves_all_live_charges_above_floor() {
        let ledger = Arc::new(Ledger::new());
        let now = Instant::now();
        let floor = 3 * 1024 * MIB;
        let charge = REQUEST_BYTES + RESPONSE_BYTES + METADATA_BYTES + 2;
        let required = floor + FIXED_BYTES + charge;
        assert!(ledger.admit(now, 1, 1, required - 1, floor).is_none());
        assert_eq!(ledger.0.lock().unwrap().active, 0);
        let first = ledger.admit(now, 1, 1, required, floor).unwrap();
        assert!(ledger.admit(now, 1, 1, required, floor).is_none());
        assert!(ledger.admit(now, 1, 1, u64::MAX, u64::MAX).is_none());
        drop(first);
        assert!(ledger.admit(now, 1, 1, required, floor).is_some());
    }
}
