//! Tiny socket pairs pin the error envelope without starting a service or model.
use super::error;
use std::{
    io::Read,
    net::{TcpListener, TcpStream},
    time::Duration,
};

fn response(status: u16, message: &str) -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let mut client =
        TcpStream::connect_timeout(&listener.local_addr().unwrap(), Duration::from_secs(3))
            .unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let (socket, _) = listener.accept().unwrap();
    error(socket, status, message);
    let mut bytes = Vec::new();
    client.read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() < 2048, "native error tiny response bound");
    String::from_utf8(bytes).unwrap()
}

#[test]
fn exact_error_envelope_status_headers_and_utf8_bytes() {
    for (status, phrase) in [
        (200, "OK"),
        (202, "Accepted"),
        (400, "Bad Request"),
        (403, "Error"),
        (404, "Not Found"),
        (408, "Error"),
        (409, "Error"),
        (413, "Error"),
        (415, "Error"),
        (422, "Error"),
        (500, "Error"),
        (503, "Service Unavailable"),
        (507, "Error"),
        (599, "Error"),
    ] {
        for (message, body) in [
            ("", r#"{"api_version":1,"error":""}"#),
            (
                "Only BF16, F16 and F32 tensors are supported",
                r#"{"api_version":1,"error":"Only BF16, F16 and F32 tensors are supported"}"#,
            ),
            (
                "bad \"rule\"\n雪\\tail",
                r#"{"api_version":1,"error":"bad \"rule\"\n雪\\tail"}"#,
            ),
        ] {
            let expected = format!("HTTP/1.1 {status} {phrase}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'\r\n\r\n{body}", body.len());
            assert_eq!(
                response(status, message),
                expected,
                "native HTTP error wire status {status}"
            );
        }
    }
}

fn rejection_response(status: u16, code: &str, message: &str) -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let mut client =
        TcpStream::connect_timeout(&listener.local_addr().unwrap(), Duration::from_secs(3))
            .unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let (socket, _) = listener.accept().unwrap();
    super::reject_now(socket, status, code, message);
    let mut bytes = Vec::new();
    client.read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() < 2048, "native rejection tiny response bound");
    String::from_utf8(bytes).unwrap()
}

#[test]
fn exact_intake_rejection_envelope_status_headers_and_utf8_bytes() {
    for (status, phrase) in [
        (400, "Bad Request"),
        (408, "Error"),
        (503, "Service Unavailable"),
        (599, "Error"),
    ] {
        for (code, message, body) in [
            ("", "", r#"{"api_version":1,"code":"","error":""}"#),
            (
                "busy",
                "Retry shortly",
                r#"{"api_version":1,"code":"busy","error":"Retry shortly"}"#,
            ),
            (
                "kind\"\\雪",
                "bad \"rule\"\n雪\\tail",
                r#"{"api_version":1,"code":"kind\"\\雪","error":"bad \"rule\"\n雪\\tail"}"#,
            ),
        ] {
            let expected = format!("HTTP/1.1 {status} {phrase}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'\r\n\r\n{body}", body.len());
            assert_eq!(
                rejection_response(status, code, message),
                expected,
                "native intake rejection wire status {status}"
            );
        }
    }
}
