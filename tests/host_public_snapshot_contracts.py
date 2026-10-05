"""Public fixture-host wire snapshots using an owned tiny registry."""
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_host.common import canonical
from atlas_host.registry import Registry
from atlas_host.runtime_adapter import FixtureHost
from host_atlas import HostHandler
from host_picker_contracts import FakeRenderer
from inference_public_snapshot_contracts import MemoryTransport

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/host-public-responses.json'


def exchange(server, method, route, data=None):
    body = b'' if data is None else json.dumps(data).encode()
    headers = {'Host':'127.0.0.1:8816', 'Origin':'http://127.0.0.1:8816',
               'X-Atlas-Local':'1', 'Content-Type':'application/json',
               'Content-Length':str(len(body))}
    wire = (f'{method} {route} HTTP/1.0\r\n' +
            ''.join(f'{key}: {value}\r\n' for key, value in headers.items()) +
            '\r\n').encode() + body
    transport = MemoryTransport(wire)
    HostHandler(transport, ('127.0.0.1', 12345), server)
    head, separator, response_body = bytes(transport.response).partition(b'\r\n\r\n')
    assert separator, 'Complete HTTP response required'
    status_line, _, header_bytes = head.partition(b'\r\n')
    response_headers = BytesParser().parsebytes(header_bytes)
    assert int(response_headers['Content-Length']) == len(response_body)
    assert response_headers['Server'] == 'BaseHTTP/0.6 Python/'+platform.python_version()
    assert parsedate_to_datetime(response_headers['Date']).utcoffset().total_seconds() == 0
    return {'status_line':status_line.decode(),
            'headers':{key:value for key, value in response_headers.items()
                       if key.lower() not in ('date', 'server')},
            'body_utf8':response_body.decode()}


def public_responses():
    with tempfile.TemporaryDirectory(prefix='atlas-host-snapshot-') as directory:
        root = Path(directory)
        source = root/'fixture'
        shutil.copytree(ROOT/'fixtures/tiny-bf16', source)
        manifest = json.loads((source/'host-manifest.json').read_text())
        registry = Registry(root/'registry.json', free_bytes=lambda _path:100*1024**3)
        readers = []

        def factory(entry, _cache):
            reader = FakeRenderer(entry)
            # Backend metadata is a declared deterministic renderer double.
            # The production host computes/redacts its own public projection.
            reader.info['source_identity'] = 'b'*64
            reader.info['model_identity'] = hashlib.sha256(canonical(
                ['weight-atlas-model-v1', 'b'*64, entry['manifest']['revision']])).hexdigest()
            readers.append(reader)
            return reader

        host = FixtureHost(registry, root/'cache', factory, clock=lambda:0.)
        server = SimpleNamespace(host=host, server_port=8816)
        results = {}
        with (patch('socket.socket', side_effect=AssertionError('No live listener')),
              patch('subprocess.Popen', side_effect=AssertionError('No renderer child')),
              patch('atlas_host.runtime_adapter.secrets.token_hex',
                    side_effect=lambda size:'a'*(2*size))):
            results['empty_catalog'] = exchange(server, 'GET', '/api/models')
            model = registry.register(source, manifest, 'Public tiny fixture')['model_id']
            results['disabled_fixture_catalog'] = exchange(server, 'GET', '/api/models')
            registry.set_enabled(model, True)
            results['enabled_fixture_catalog'] = exchange(server, 'GET', '/api/models')
            results['disabled_inference'] = exchange(server, 'GET', '/api/inference')
            results['disabled_analytics'] = exchange(server, 'GET', '/api/analytics')
            results['unavailable_route'] = exchange(server, 'GET', '/api/unavailable')
            results['acquire_fixture'] = exchange(server, 'POST', '/api/view-contexts', {'model_id':model})
            lease = json.loads(results['acquire_fixture']['body_utf8'])
            context, capability = lease['context_id'], lease['capability']
            results['selected_model'] = exchange(server, 'GET', f'/api/models/{model}/model?context={context}')
            results['active_catalog'] = exchange(server, 'GET', '/api/models')
            results['heartbeat'] = exchange(server, 'POST', f'/api/view-contexts/{context}/heartbeat', {'capability':capability})
            results['release'] = exchange(server, 'POST', f'/api/view-contexts/{context}/release', {'capability':capability})
            results['stale_heartbeat'] = exchange(server, 'POST', f'/api/view-contexts/{context}/heartbeat', {'capability':capability})
        assert host.close() and len(readers) == 1 and not readers[0].running
        return results


class PublicSnapshots(unittest.TestCase):
    def test_exact_public_responses_and_same_run_determinism(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture['excluded_headers'], ['Date', 'Server'])
        first, second = public_responses(), public_responses()
        self.assertEqual(first, second)
        self.assertEqual(first, fixture['responses'])


if __name__ == '__main__':
    unittest.main()
