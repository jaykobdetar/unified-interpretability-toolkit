use super::{comparison, viewer, VIEWER_SCRIPTS};

#[test]
fn viewer_files_keep_exact_mime_and_bytes() {
    let expected: &[(&str, &str, &[u8])] = &[
        (
            "/",
            "text/html; charset=utf-8",
            include_bytes!("../web/index.html"),
        ),
        (
            "/index.html",
            "text/html; charset=utf-8",
            include_bytes!("../web/index.html"),
        ),
        (
            "/inference.js",
            "text/javascript",
            include_bytes!("../web/inference.js"),
        ),
        (
            "/inference-import.js",
            "text/javascript",
            include_bytes!("../web/inference-import.js"),
        ),
        (
            "/atlas-tools.js",
            "text/javascript",
            include_bytes!("../web/atlas-tools.js"),
        ),
        (
            "/workspace-tools.js",
            "text/javascript",
            include_bytes!("../web/workspace-tools.js"),
        ),
        (
            "/app.js",
            "text/javascript",
            include_bytes!("../web/app.js"),
        ),
        ("/style.css", "text/css", include_bytes!("../web/style.css")),
        (
            "/vendor/openseadragon.min.js",
            "text/javascript",
            include_bytes!("../web/vendor/openseadragon.min.js"),
        ),
        (
            "/vendor/OpenSeadragon-LICENSE.txt",
            "text/plain",
            include_bytes!("../web/vendor/OpenSeadragon-LICENSE.txt"),
        ),
    ];
    for &(path, mime, bytes) in expected {
        assert_eq!(viewer(path), Some((mime, bytes)), "{path}");
    }
    for path in [
        "/comparison.html",
        "/comparison.js",
        "/host-client.js",
        "/missing-fixture.js",
    ] {
        assert!(viewer(path).is_none(), "{path}");
    }
}

#[test]
fn comparison_files_keep_exact_mime_bytes_and_viewer_separation() {
    let expected: &[(&str, &str, &[u8])] = &[
        (
            "/",
            "text/html; charset=utf-8",
            include_bytes!("../web/comparison.html"),
        ),
        (
            "/comparison.html",
            "text/html; charset=utf-8",
            include_bytes!("../web/comparison.html"),
        ),
        (
            "/comparison.js",
            "text/javascript",
            include_bytes!("../web/comparison.js"),
        ),
        (
            "/comparison.css",
            "text/css",
            include_bytes!("../web/comparison.css"),
        ),
        (
            "/vendor/openseadragon.min.js",
            "text/javascript",
            include_bytes!("../web/vendor/openseadragon.min.js"),
        ),
        (
            "/vendor/OpenSeadragon-LICENSE.txt",
            "text/plain",
            include_bytes!("../web/vendor/OpenSeadragon-LICENSE.txt"),
        ),
    ];
    for &(path, mime, bytes) in expected {
        assert_eq!(comparison(path), Some((mime, bytes)), "{path}");
    }
    for path in [
        "/index.html",
        "/app.js",
        "/inference.js",
        "/missing-fixture.js",
    ] {
        assert!(comparison(path).is_none(), "{path}");
    }
}

#[test]
fn startup_has_exact_order_and_separators() {
    let expected: [&[u8]; 9] = [
        include_bytes!("../web/vendor/openseadragon.min.js"),
        b"\n;\n",
        include_bytes!("../web/atlas-tools.js"),
        b"\n;\n",
        include_bytes!("../web/app.js"),
        b"\n;\n",
        include_bytes!("../web/workspace-tools.js"),
        b"\n;\n",
        include_bytes!("../web/inference.js"),
    ];
    assert_eq!(VIEWER_SCRIPTS, expected);
}
