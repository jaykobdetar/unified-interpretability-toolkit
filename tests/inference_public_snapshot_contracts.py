"""Exact public HTTP responses through the real handler; transport only is fake."""
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import io
import json
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import live_inference as live

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/inference-public-responses.json'


class MemoryTransport:
    def __init__(self, wire):
        self.wire = wire
        self.response = bytearray()

    def makefile(self, *_args):
        return io.BytesIO()

    def settimeout(self, _seconds):
        pass

    def recv(self, size):
        block, self.wire = self.wire[:size], self.wire[size:]
        return block

    def sendall(self, block):
        self.response.extend(block)


def public_responses():
    with tempfile.TemporaryDirectory(prefix='atlas-response-snapshot-') as directory:
        return _responses_for_empty_model(Path(directory))


def _responses_for_empty_model(directory):
    session = live.Session('unused-python', directory)
    server = SimpleNamespace(session=session, server_port=8816)
    scenarios = [('idle_metadata', 'GET', '/api/inference', None),
                 ('poll_without_active_owner', 'POST', '/api/inference/poll',
                  {'session':'public-synthetic-owner'}),
                 ('token_limit_boundary_without_model', 'POST', '/api/inference/start',
                  {'prompt':'Public synthetic fixture', 'max_new_tokens':32, 'layer':0})]
    results = {}
    for name, method, route, data in scenarios:
        body = b'' if data is None else json.dumps(data).encode()
        headers = {'Host':'127.0.0.1:8816', 'Origin':'http://127.0.0.1:8816',
                   'X-Atlas-Local':'1', 'Content-Type':'application/json',
                   'Content-Length':str(len(body))}
        wire = (f'{method} {route} HTTP/1.0\r\n' +
                ''.join(f'{key}: {value}\r\n' for key, value in headers.items()) +
                '\r\n').encode() + body
        transport = MemoryTransport(wire)
        # Fixed admission environment; the unavailable model then refuses before
        # any child can start. This exposes the token boundary without inference.
        with patch.object(live, 'available', return_value=8*live.GIB):
            live.Handler(transport, ('127.0.0.1', 12345), server)
        head, separator, response_body = bytes(transport.response).partition(b'\r\n\r\n')
        assert separator, 'Complete HTTP response required'
        status_line, _, header_bytes = head.partition(b'\r\n')
        response_headers = BytesParser().parsebytes(header_bytes)
        assert int(response_headers['Content-Length']) == len(response_body)
        assert response_headers['Server'] == 'BaseHTTP/0.6 Python/'+platform.python_version()
        assert parsedate_to_datetime(response_headers['Date']).utcoffset().total_seconds() == 0
        # Date is wall clock; Server describes the selected Python runtime.
        # Keep all application headers and exact response body bytes.
        stable_headers = {key:value for key, value in response_headers.items()
                          if key.lower() not in ('date', 'server')}
        results[name] = {'status_line':status_line.decode(), 'headers':stable_headers,
                         'body_utf8':response_body.decode()}
    return results


class PublicSnapshots(unittest.TestCase):
    def test_exact_public_responses_and_same_run_determinism(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture['excluded_headers'], ['Date', 'Server'])
        with (patch('socket.socket', side_effect=AssertionError('No live listener')),
              patch('subprocess.Popen', side_effect=AssertionError('No model child'))):
            first = public_responses()
            second = public_responses()
        self.assertEqual(first, second)
        self.assertEqual(first, fixture['responses'])


if __name__ == '__main__':
    unittest.main()
