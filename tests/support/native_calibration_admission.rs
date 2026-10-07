//! Pin calibration queue admission, retained errors and exact HTTP replies.
use super::{reply_calibration, Job, Query};
use crate::{render::Stats, state::State};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::Read,
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{sync_channel, SyncSender},
    },
    time::Duration,
};

struct Fixture {
    root: PathBuf,
    state: State,
}
impl Fixture {
    fn new(dtype: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-calibration-admission-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let payload = if dtype == "BF16" {
            0x7fc0u16.to_le_bytes().to_vec()
        } else {
            vec![0]
        };
        let header =
            json!({"nan.weight":{"dtype":dtype,"shape":[1,1],"data_offsets":[0,payload.len()]}})
                .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(model.join("tiny.safetensors"), bytes).unwrap();
        let state = State::open(&model, &root.join("cache"), None, None).unwrap();
        state.progress.lock().unwrap().error = Some("prior refusal".into());
        Self { root, state }
    }
    fn ready(&self) {
        self.state.calibration.lock().unwrap().tensors.insert(
            0,
            Stats {
                count: 1,
                max_abs: 1.0,
                median_nonzero_abs: 1.0,
                q99: 1.0,
                robust_clipped_count: 0,
                histogram_sha256: None,
                exact_zero_count: 0,
                unique_bit_patterns: Some(1),
                dtype: "BF16".into(),
                calibration_method: "held admission stats".into(),
                seconds: 0.0,
            },
        );
    }
    fn response(
        &self,
        sender: &SyncSender<Job>,
        fields: &[(&str, &str)],
        local: Option<&str>,
    ) -> String {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let mut client =
            TcpStream::connect_timeout(&listener.local_addr().unwrap(), Duration::from_secs(3))
                .unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let (socket, _) = listener.accept().unwrap();
        drop(listener);
        let q = Query(
            fields
                .iter()
                .map(|(k, v)| ((*k).into(), (*v).into()))
                .collect(),
        );
        let mut headers = BTreeMap::new();
        if let Some(value) = local {
            headers.insert("x-atlas-local".into(), value.into());
        }
        reply_calibration(&self.state, sender, socket, &q, &headers);
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).unwrap();
        assert!(bytes.len() < 32 * 1024, "calibration tiny response bound");
        String::from_utf8(bytes).unwrap()
    }
    fn prior_error(&self) {
        assert_eq!(
            self.state.progress.lock().unwrap().error.as_deref(),
            Some("prior refusal"),
            "calibration refusal retains prior error"
        );
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn wire(status: u16, phrase: &str, value: Value) -> String {
    let body = value.to_string();
    format!("HTTP/1.1 {status} {phrase}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'\r\n\r\n{body}", body.len())
}

fn refusal(status: u16, message: &str) -> String {
    wire(
        status,
        if status == 400 {
            "Bad Request"
        } else {
            "Service Unavailable"
        },
        json!({"api_version":1,"error":message}),
    )
}
#[test]
fn admission_replies_and_progress_are_exact() {
    let _workspace = crate::resources::test_workspace_guard();
    for local in [None, Some("2")] {
        let f = Fixture::new("BF16");
        let (tx, rx) = sync_channel(1);
        assert_eq!(
            f.response(&tx, &[("tensor", "bad")], local),
            refusal(400, "Local action header required"),
            "calibration local header"
        );
        assert!(rx.try_recv().is_err(), "calibration header queues nothing");
        f.prior_error();
    }
    for full in [false, true] {
        let f = Fixture::new("BF16");
        let (tx, rx) = sync_channel(1);
        if full {
            assert!(tx.try_send(Job::Wake).is_ok());
        }
        assert_eq!(
            f.response(&tx, &[("all", "1")], Some("1")),
            wire(
                202,
                "Accepted",
                json!({"api_version":1,"queued":"complete checkpoint"})
            ),
            "calibration all reply"
        );
        let p = f.state.progress.lock().unwrap();
        assert!(p.all_requested, "calibration all requested assertion");
        assert_eq!(p.error, None, "calibration all clears prior error");
        assert!(
            matches!(rx.try_recv(), Ok(Job::Wake)),
            "calibration all wake"
        );
    }
    {
        let f = Fixture::new("BF16");
        let (tx, rx) = sync_channel(1);
        drop(rx);
        assert_eq!(
            f.response(&tx, &[("all", "1")], Some("1")),
            refusal(503, "Numeric worker unavailable"),
            "calibration all disconnected"
        );
        assert!(
            !f.state.progress.lock().unwrap().all_requested,
            "calibration disconnected all flag"
        );
        f.prior_error();
    }
    {
        let f = Fixture::new("BF16");
        let (tx, rx) = sync_channel(1);
        assert_eq!(
            f.response(&tx, &[], Some("1")),
            wire(202, "Accepted", json!({"api_version":1,"queued":0})),
            "calibration default single reply"
        );
        assert!(
            matches!(rx.try_recv(), Ok(Job::Calibrate(0))),
            "calibration default single job"
        );
        assert_eq!(
            f.state.progress.lock().unwrap().error,
            None,
            "calibration single clears prior error"
        );
    }
    {
        let f = Fixture::new("BF16");
        f.ready();
        let (tx, rx) = sync_channel(1);
        assert_eq!(
            f.response(&tx, &[], Some("1")),
            wire(200, "OK", json!({"api_version":1,"complete":true})),
            "calibration cached reply"
        );
        assert!(rx.try_recv().is_err(), "calibration cached queues nothing");
        f.prior_error();
    }
    for disconnected in [false, true] {
        let f = Fixture::new("BF16");
        let (tx, rx) = sync_channel(1);
        let receiver = if disconnected {
            drop(rx);
            None
        } else {
            assert!(tx.try_send(Job::Wake).is_ok());
            Some(rx)
        };
        assert_eq!(
            f.response(&tx, &[], Some("1")),
            refusal(503, "Numeric queue full; retry shortly"),
            "calibration single refusal"
        );
        f.prior_error();
        drop(receiver);
    }
    for (dtype, tensor, message) in [
        ("BF16", "99", "Unknown tensor"),
        ("BF16", "bad", "invalid digit found in string"),
        (
            "I8",
            "0",
            "Storage extent validated; numeric encoding/quantization scale semantics unsupported",
        ),
    ] {
        let f = Fixture::new(dtype);
        let (tx, rx) = sync_channel(1);
        assert_eq!(
            f.response(&tx, &[("tensor", tensor)], Some("1")),
            refusal(400, message),
            "calibration invalid tensor reply"
        );
        assert!(
            rx.try_recv().is_err(),
            "calibration invalid tensor queues nothing"
        );
        f.prior_error();
    }
}

#[test]
fn publication_lock_covers_admission_and_error_clear() {
    let _workspace = crate::resources::test_workspace_guard();
    let fixture = Fixture::new("BF16");
    for accepted in [false, true] {
        fixture.state.progress.lock().unwrap().error = Some("prior refusal".into());
        let mut calls = 0;
        let result = super::queue_calibration(&fixture.state, || {
            calls += 1;
            assert!(
                matches!(
                    fixture.state.progress.try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ),
                "calibration publication lock assertion held throughout admission"
            );
            accepted
        });
        assert_eq!(calls, 1, "calibration admission invoked once");
        assert_eq!(result, accepted, "calibration admission return value");
        assert_eq!(
            fixture.state.progress.lock().unwrap().error.as_deref(),
            if accepted {
                None
            } else {
                Some("prior refusal")
            },
            "calibration success clears and refusal preserves error"
        );
    }
}

#[test]
fn failed_worker_publishes_after_accepted_admission() {
    let _workspace = crate::resources::test_workspace_guard();
    let fixture = Fixture::new("BF16");
    std::thread::scope(|scope| {
        let (entered, entry) = sync_channel(1);
        let state = &fixture.state;
        let mut worker = None;
        assert!(
            super::queue_calibration(state, || {
                assert!(
                    matches!(
                        state.progress.try_lock(),
                        Err(std::sync::TryLockError::WouldBlock)
                    ),
                    "calibration publication lock assertion excludes worker during admission"
                );
                worker = Some(scope.spawn(move || {
                    entered.send(()).unwrap();
                    state.calibrate_one(0).unwrap_err().to_string()
                }));
                entry.recv_timeout(Duration::from_secs(2)).unwrap();
                true
            }),
            "calibration worker admission accepted"
        );
        let error = worker.unwrap().join().unwrap();
        assert_eq!(
            state.progress.lock().unwrap().error.as_deref(),
            Some(error.as_str()),
            "calibration worker error survives admission completion"
        );
        assert!(
            state.stats(0).is_none(),
            "calibration failure does not install statistics"
        );
        assert_eq!(
            state.status(Some(0)).unwrap()["coverage"]["calibration_error"],
            "calibration_failed",
            "calibration public status retains failed worker result"
        );
    });
}
