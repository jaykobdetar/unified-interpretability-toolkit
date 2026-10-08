//! Pin the view envelope and refusal bytes using tiny headers and in-memory stats.
//! Identity/legend helpers retain their separate qualification; this pins their
//! placement and caller arguments, not an independent numeric or identity oracle.
use super::{reply_view, Query};
use crate::{render::Stats, slice::TensorSlice, state::State};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

struct Fixture {
    root: PathBuf,
    state: State,
    header_len: usize,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-view-response-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let header =
            json!({"volume-λ":{"dtype":"BF16","shape":[2,1,2],"data_offsets":[0,8]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0; 8]);
        std::fs::write(model.join("tiny.safetensors"), bytes).unwrap();
        let state = State::open(&model, &root.join("cache"), None, Some("view/λ".into())).unwrap();
        Self {
            root,
            state,
            header_len: header.len(),
        }
    }

    fn ready(&self) {
        self.state.calibration.lock().unwrap().tensors.insert(
            0,
            Stats {
                count: 4,
                max_abs: 5.25,
                median_nonzero_abs: 0.75,
                q99: 4.0,
                robust_clipped_count: 1,
                histogram_sha256: None,
                exact_zero_count: 1,
                unique_bit_patterns: Some(4),
                dtype: "BF16".into(),
                calibration_method: "held view presentation stats".into(),
                seconds: 0.0,
            },
        );
    }

    fn response(&self, fields: &[(&str, &str)]) -> String {
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
                .collect::<BTreeMap<_, _>>(),
        );
        reply_view(&self.state, socket, &q);
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).unwrap();
        assert!(bytes.len() < 32 * 1024, "view tiny response bound");
        String::from_utf8(bytes).unwrap()
    }

    fn expected(&self, leading: usize, left: &str, right: &str) -> Value {
        let slice = TensorSlice::new(&self.state.source, 0, &[leading]).unwrap();
        let legends = self
            .state
            .legends(self.state.source.tensor(0).unwrap(), left, right)
            .unwrap();
        json!({
            "api_version":1,
            "tensor":{
                "id":0,"name":"volume-λ","shape":[2,1,2],"rows":1,"cols":2,
                "count":4,"dtype":"BF16","element_bytes":2,"available":true,
                "unavailable_reason":null,"shard":"tiny.safetensors","shard_id":0,
                "byte_offset":self.header_len + 8,"max_level":1,"min_level":0,
                "slice":[leading],"slice_identity":slice.identity,"slice_count":2
            },
            "source_binding":self.state.slice_binding(&slice),
            "legends":legends,
            "tile_bindings":{
                "left":self.state.tile_binding_for(&slice,left,&legends["left"]),
                "right":self.state.tile_binding_for(&slice,right,&legends["right"])
            },
            "tile_size":256,"overlap":0,"source_values_unchanged":true
        })
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

#[test]
fn full_view_envelope_and_refusals_preserve_bytes() {
    let _workspace = crate::resources::test_workspace_guard();
    let fixture = Fixture::new();
    assert_eq!(
        fixture.response(&[("slice", "1")]),
        wire(
            503,
            "Service Unavailable",
            json!({"api_version":1,"error":"Full checkpoint calibration is not ready"})
        ),
        "view pending refusal"
    );
    assert_eq!(
        fixture.response(&[("tensor", "99")]),
        wire(
            400,
            "Bad Request",
            json!({"api_version":1,"error":"Unknown tensor"})
        ),
        "view invalid tensor refusal"
    );
    assert_eq!(
        fixture.response(&[]),
        wire(
            400,
            "Bad Request",
            json!({"api_version":1,"error":"Every leading axis requires an explicit slice index"})
        ),
        "view missing slice refusal"
    );
    fixture.ready();
    assert_eq!(
        fixture.response(&[("slice", "1")]),
        wire(
            200,
            "OK",
            fixture.expected(1, "global_linear", "global_asinh")
        ),
        "view default envelope"
    );
    assert_eq!(
        fixture.response(&[
            ("tensor", "0"),
            ("slice", "0"),
            ("left", "tensor_asinh"),
            ("right", "tensor_linear")
        ]),
        wire(
            200,
            "OK",
            fixture.expected(0, "tensor_asinh", "tensor_linear")
        ),
        "view explicit envelope"
    );
    std::fs::OpenOptions::new()
        .append(true)
        .open(fixture.root.join("model/tiny.safetensors"))
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert_eq!(
        fixture.response(&[("tensor", "bad")]),
        wire(
            400,
            "Bad Request",
            json!({"api_version":1,"error":"Source identity changed; stop and reopen model. Cached calibration/tiles are invalid."})
        ),
        "view source refusal precedes query parsing"
    );
}
