"""Profile HTTP snapshots with the actual router/service and existing doubles."""
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import platform
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_host.profile_http import HostedHandler
from atlas_host.registry import Registry
from atlas_host.runtime_adapter import FixtureHost
from inference_public_snapshot_contracts import MemoryTransport
import profile_runtime_doubles as doubles

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/profile-public-responses.json'


def exchange(server, method, route, data=None):
    body = b'' if data is None else json.dumps(data).encode()
    headers = {'Host':'127.0.0.1:8816', 'Origin':'http://127.0.0.1:8816',
               'X-Atlas-Local':'1', 'Content-Type':'application/json',
               'Content-Length':str(len(body))}
    wire = (f'{method} {route} HTTP/1.0\r\n' +
            ''.join(f'{key}: {value}\r\n' for key, value in headers.items()) +
            '\r\n').encode() + body
    transport = MemoryTransport(wire)
    HostedHandler(transport, ('127.0.0.1', 12345), server)
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
    case = doubles.ServiceTests('test_second_start_never_queues')
    case.setUp()
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-profile-snapshot-') as directory:
            root = Path(directory)
            host = FixtureHost(Registry(root/'registry.json'), root/'cache',
                               lambda *_args: (_ for _ in ()).throw(AssertionError('No renderer')),
                               clock=lambda:0.)
            # The HTTP supervision-context admission bridge is doubled. The
            # real service/router own grants, validation, state and cleanup.
            bridge = SimpleNamespace(
                admission=lambda:(case.clock.grant(), doubles.Meter()),
                finish_admission=case.service.finish_admission)
            app = SimpleNamespace(profiles=bridge, api=case.api, host=host,
                                  supervisor=case.service.supervisor, lock=threading.RLock())
            server = SimpleNamespace(application=app, host=host, server_port=8816,
                                     profile_controls=False)
            counters = {16:0, 32:0}
            def entropy(size):
                counters[size] += 1
                digits = '1234' if size == 16 else '5678'
                return digits[(counters[size]-1) % len(digits)]*(2*size)
            results = {}
            with (patch('socket.socket', side_effect=AssertionError('No live listener')),
                  patch('subprocess.Popen', side_effect=AssertionError('No live child')),
                  patch('threading.Thread.start', side_effect=AssertionError('No live thread')),
                  patch('secrets.token_hex', side_effect=entropy)):
                results['disabled_catalog'] = exchange(server, 'GET', '/api/models')
                results['disabled_start'] = exchange(server, 'POST', '/api/profiles/start', case.data)
                server.profile_controls = True
                results['owner_catalog'] = exchange(server, 'GET', '/api/models')
                full_data = {**case.data, 'values':12}
                results['full_start'] = exchange(server, 'POST', '/api/profiles/start', full_data)
                start = json.loads(results['full_start']['body_utf8'])
                owner = {key:case.data[key] for key in ('version', 'model_id', 'context_id', 'tab_capability')}
                owner.update(job_id=start['job_id'], job_capability=start['job_capability'])
                results['admitting_status'] = exchange(server, 'POST', '/api/profiles/status', owner)
                assert not hasattr(case, 'hooks'), 'HTTP status must not drive execution'
                case.service.step(); case.finish()
                results['complete_status'] = exchange(server, 'POST', '/api/profiles/status', owner)
                complete = json.loads(results['complete_status']['body_utf8'])
                for axis, count in [('rows',3), ('columns',4)]:
                    results['complete_'+axis] = exchange(server, 'POST', '/api/profiles/page',
                        {**owner, 'revision':complete['accepted']['revision'], 'axis':axis, 'start':0, 'count':count})
                results['complete_heartbeat'] = exchange(server, 'POST', '/api/profiles/heartbeat', owner)
                results['partial_restart'] = exchange(server, 'POST', '/api/profiles/start',
                    {**case.data, 'values':5, 'restart':True})
                restarted = json.loads(results['partial_restart']['body_utf8'])
                results['stale_status'] = exchange(server, 'POST', '/api/profiles/status', owner)
                owner.update(job_id=restarted['job_id'], job_capability=restarted['job_capability'])
                case.service.step(); case.finish()
                results['partial_status'] = exchange(server, 'POST', '/api/profiles/status', owner)
                partial = json.loads(results['partial_status']['body_utf8'])
                results['partial_rows'] = exchange(server, 'POST', '/api/profiles/page',
                    {**owner, 'revision':partial['accepted']['revision'], 'axis':'rows', 'start':0, 'count':3})
                results['cancel_pending'] = exchange(server, 'POST', '/api/profiles/cancel', owner)
                case.service.step()
                results['cancelled_status'] = exchange(server, 'POST', '/api/profiles/status', owner)
                results['reconciled'] = exchange(server, 'POST', '/api/profiles/reconcile',
                    {key:case.data[key] for key in ('version', 'model_id', 'context_id', 'tab_capability')})
            assert not case.os.files and case.service.supervisor.charged_snapshot_bytes() == 0
            assert not case.service.supervisor.busy() and host.close()
            return results
    finally:
        case.doCleanups()


class PublicSnapshots(unittest.TestCase):
    def test_exact_public_responses_and_same_run_determinism(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture['excluded_headers'], ['Date', 'Server'])
        first, second = public_responses(), public_responses()
        self.assertEqual(first, second)
        self.assertEqual(first, fixture['responses'])


if __name__ == '__main__':
    unittest.main()
