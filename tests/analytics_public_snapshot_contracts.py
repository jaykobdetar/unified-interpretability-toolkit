"""Public analytics HTTP snapshots with real tiny-source arithmetic."""

from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import platform
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics.service import AnalyticsJobs
from analytics.worker import analyze_request
from inference_public_snapshot_contracts import MemoryTransport
import live_inference as live

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/analytics-public-responses.json"


def exchange(server, method, route, data=None):
    body = b"" if data is None else json.dumps(data).encode()
    headers = {
        "Host": "127.0.0.1:8816",
        "Origin": "http://127.0.0.1:8816",
        "X-Atlas-Local": "1",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    wire = (
        f"{method} {route} HTTP/1.0\r\n"
        + "".join(f"{key}: {value}\r\n" for key, value in headers.items())
        + "\r\n"
    ).encode() + body
    transport = MemoryTransport(wire)
    live.Handler(transport, ("127.0.0.1", 12345), server)
    head, separator, response_body = bytes(transport.response).partition(b"\r\n\r\n")
    assert separator, "Complete HTTP response required"
    status_line, _, header_bytes = head.partition(b"\r\n")
    response_headers = BytesParser().parsebytes(header_bytes)
    assert int(response_headers["Content-Length"]) == len(response_body)
    assert (
        response_headers["Server"] == "BaseHTTP/0.6 Python/" + platform.python_version()
    )
    assert (
        parsedate_to_datetime(response_headers["Date"]).utcoffset().total_seconds() == 0
    )
    return {
        "status_line": status_line.decode(),
        "headers": {
            key: value
            for key, value in response_headers.items()
            if key.lower() not in ("date", "server")
        },
        "body_utf8": response_body.decode(),
    }


def public_responses():
    with tempfile.TemporaryDirectory(prefix="atlas-analytics-snapshot-") as directory:
        root = Path(directory)
        header = json.dumps(
            {"fixture": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}
        ).encode()
        path = root / "tiny.safetensors"
        values = [1.0, -2.0, 0.0, 4.0, 5.0, -6.0]
        path.write_bytes(
            struct.pack("<Q", len(header))
            + header
            + b"".join(
                struct.pack(
                    "<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16
                )
                for value in values
            )
        )
        tensor = {
            "id": 0,
            "name": "fixture",
            "dtype": "BF16",
            "shape": [2, 3],
            "shard": path.name,
            "byte_offset": 8 + len(header),
        }
        model = {
            "source_directory": str(root),
            "source_identity": "trusted-fixture",
            "catalog": [tensor],
        }
        data = {
            "tensor": 0,
            "region": {"row": 0, "col": 0, "rows": 2, "cols": 3},
            "seed": 1,
            "svd": False,
        }
        state = {"inference_busy": False, "drain": False, "event": b""}
        jobs = AnalyticsJobs(
            "unused-python",
            root,
            inference_busy=lambda: state["inference_busy"],
            fetch_model=lambda: model,
            available=lambda: 8 * live.GIB,
            reap=lambda _process: True,
            clock=lambda: 10.0,
        )
        session = live.Session("unused-python", root)
        session.analytics = jobs
        server = SimpleNamespace(session=session, analytics=jobs, server_port=8816)

        def spawn(*_args, **kwargs):
            # Only process ownership is doubled. The actual source/header checks,
            # raw BF16 reads and arithmetic produce the worker protocol event.
            payload = json.loads(kwargs["stdin"].read())
            result = analyze_request(payload)
            state["event"] = json.dumps({"ok": True, "result": result}).encode() + b"\n"
            return SimpleNamespace(
                pid=-999,
                stdout=SimpleNamespace(fileno=lambda: -999, close=lambda: None),
            )

        original_read = __import__("os").read

        def read(fd, size):
            if fd != -999:
                return original_read(fd, size)
            if not state["drain"]:
                raise BlockingIOError
            chunk, state["event"] = state["event"][:size], state["event"][size:]
            return chunk

        results = {}
        with (
            patch("socket.socket", side_effect=AssertionError("No live listener")),
            # Filesystem identity fields vary across temporary directories.
            # Supply fixed stat provenance, retaining the actual byte length;
            # source/cache identities are then computed and pinned unchanged.
            patch(
                "analytics.source.fingerprint",
                side_effect=lambda st: (101, 202, st.st_size, 303, 404),
            ),
            patch(
                "analytics.worker.fingerprint",
                side_effect=lambda st: (101, 202, st.st_size, 303, 404),
            ),
            patch("analytics.service.subprocess.Popen", side_effect=spawn),
            patch("analytics.service.os.set_blocking"),
            patch("analytics.service.os.read", side_effect=read),
            patch("analytics.service.secrets.token_hex", return_value="a" * 32),
            patch(
                "analytics.service.shutil.disk_usage",
                return_value=SimpleNamespace(free=30 * live.GIB),
            ),
        ):
            results["idle_metadata"] = exchange(server, "GET", "/api/analytics")
            state["inference_busy"] = True
            results["inference_busy_metadata"] = exchange(
                server, "GET", "/api/analytics"
            )
            results["inference_busy_start"] = exchange(
                server, "POST", "/api/analytics/start", data
            )
            state["inference_busy"] = False
            results["poll_without_owner"] = exchange(
                server, "POST", "/api/analytics/poll", {"job": "other"}
            )
            results["running_start"] = exchange(
                server, "POST", "/api/analytics/start", data
            )
            state["drain"] = True
            results["complete_poll"] = exchange(
                server, "POST", "/api/analytics/poll", {"job": "a" * 32}
            )
            state["drain"] = False
            results["second_running_start"] = exchange(
                server, "POST", "/api/analytics/start", data
            )
            results["cancelled"] = exchange(
                server, "POST", "/api/analytics/cancel", {"job": "a" * 32}
            )
        assert jobs.process is None and not jobs.busy
        return results


class PublicSnapshots(unittest.TestCase):
    def test_exact_public_responses_and_same_run_determinism(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture["excluded_headers"], ["Date", "Server"])
        first, second = public_responses(), public_responses()
        self.assertEqual(first, second)
        self.assertEqual(first, fixture["responses"])


if __name__ == "__main__":
    unittest.main()
