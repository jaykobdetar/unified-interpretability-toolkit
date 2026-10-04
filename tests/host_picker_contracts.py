"""Pure fixture-route and lifecycle tests. No process, socket or renderer starts."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from atlas_host.common import canonical
from atlas_host.registry import Registry
from atlas_host.runtime_adapter import FixtureHost, HostError, dispatch
from atlas_host.prepare_fixture import prepare
from host_atlas import HostHandler, Renderer, GIB

MANIFEST = json.loads((ROOT/'fixtures/tiny-bf16/host-manifest.json').read_text())


class FakeRenderer:
    def __init__(self, entry):
        self.entry = entry
        self.running, self.ready_flag = True, True
        self.stops = 0
        self.stop_result = True
        self.requests = []
        source = hashlib.sha256(entry['root'].encode()).hexdigest()
        model = hashlib.sha256(canonical(['weight-atlas-model-v1', source,
                                         entry['manifest']['revision']])).hexdigest()
        self.info = {'api_version': 1, 'source_directory': entry['root'],
                     'source_identity': source, 'model_identity': model,
                     'revision': entry['manifest']['revision'], 'source_bytes': 244, 'parameter_count': 26,
                     'inference_source_model': {'should': 'be removed'}, 'head_layout': {'bad': 'upstream'},
                     'head_layout_binding': {'bad': 'upstream'},
                     'catalog': [{'id': i, 'name': name, 'shape': shape, 'dtype': 'BF16',
                                  'shard': 'tiny.safetensors'} for i, (name, shape) in
                                 enumerate([('matrix', [3, 5]), ('vector', [7]), ('zeros', [4])])]}

    def ready(self):
        return self.ready_flag

    def alive(self):
        return self.running

    def stop(self):
        self.stops += 1
        if self.stop_result:
            self.running = False
        return self.stop_result

    def read(self, path):
        self.requests.append(path)
        if path == '/api/model':
            return 200, canonical(self.info), 'application/json'
        if path.startswith(('/api/progress', '/api/tensor-status')):
            body = {key: self.info[key] for key in ('api_version', 'source_identity', 'model_identity')}
            body.update(model_status_version=1, calibration_complete=False, global_max=None,
                        coverage={'calibrated_tensors': 0, 'values_streamed': 0, 'active_tensor': None,
                                  'all_requested': False, 'calibration_error': None},
                        tensor_status={'id': 0, 'calibration_complete': False})
            return 200, canonical(body), 'application/json'
        return 200, canonical({'api_version': 1}), 'application/json'


class HostFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = Registry(self.root/'registry.json', free_bytes=lambda _: 100*1024**3)
        self.ids, self.sources = [], []
        for label in ('first', 'second'):
            source = self.root/label
            shutil.copytree(ROOT/'fixtures/tiny-bf16', source)
            manifest = {**deepcopy(MANIFEST), 'repository': 'weight-atlas/'+label}
            identifier = self.registry.register(source, manifest, label)['model_id']
            self.registry.set_enabled(identifier, True)
            self.ids.append(identifier)
            self.sources.append(source)
        self.now = 0.0
        self.readers = []
        def factory(entry, cache):
            self.assertFalse(cache.is_relative_to(Path(entry['root'])))
            reader = FakeRenderer(entry)
            self.readers.append(reader)
            return reader
        self.host = FixtureHost(self.registry, self.root/'cache', factory, clock=lambda: self.now)

    def acquire(self, index=0, **current):
        status, raw, _ = dispatch(self.host, 'POST', '/api/view-contexts', {'model_id': self.ids[index], **current})
        self.assertEqual(status, 202)
        return json.loads(raw)

    def read(self, lease, route='model', extra=''):
        status, raw, mime = dispatch(self.host, 'GET',
            f'/api/models/{lease["model_id"]}/{route}?context={lease["context_id"]}'+extra)
        self.assertEqual(status, 200)
        return json.loads(raw)

    def test_catalog_is_read_only_and_does_not_activate_or_hash_payloads(self):
        with patch('atlas_host.runtime_adapter.check_fixture', side_effect=AssertionError('no payload read')):
            status, raw, _ = dispatch(self.host, 'GET', '/api/models')
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(raw)['models']), 2)
        self.assertEqual(self.readers, [])
        self.assertNotIn(str(self.root), raw.decode())
        self.assertFalse(json.loads(raw)['inference_enabled'])

    def test_activation_binds_source_and_redacts_trusted_internal_metadata(self):
        lease = self.acquire()
        model = self.read(lease)
        self.assertNotIn('source_directory', model)
        self.assertNotIn('inference_source_model', model)
        self.assertNotIn('head_layout', model)
        self.assertNotIn('head_layout_binding', model)
        self.assertFalse(model['inference_editable'])
        self.assertEqual(model['host_context']['context_id'], lease['context_id'])
        self.assertNotIn(lease['capability'], json.dumps(model))

    def test_shared_reader_and_cross_tab_busy_preserve_other_lease(self):
        first, second = self.acquire(), self.acquire()
        self.assertEqual(first['context_id'], second['context_id'])
        self.assertNotEqual(first['capability'], second['capability'])
        with self.assertRaises(HostError) as error:
            self.acquire(1, context_id=first['context_id'], capability=first['capability'])
        self.assertEqual(error.exception.code, 'reader_busy')
        self.assertEqual(len(self.readers), 1)
        self.assertEqual(self.readers[0].stops, 0)
        self.read(second)

    def test_owned_switch_reaps_old_reader_and_rejects_old_context(self):
        first = self.acquire()
        self.read(first)
        second = self.acquire(1, context_id=first['context_id'], capability=first['capability'])
        self.assertNotEqual(first['context_id'], second['context_id'])
        self.assertFalse(self.readers[0].running)
        with self.assertRaises(HostError) as error:
            self.read(first)
        self.assertEqual(error.exception.code, 'stale_context')
        self.read(second)

    def test_release_and_heartbeat_require_owned_unexpired_capability(self):
        first, second = self.acquire(), self.acquire()
        for action in ('release', 'heartbeat'):
            with self.assertRaises(HostError):
                dispatch(self.host, 'POST', f'/api/view-contexts/{first["context_id"]}/{action}', {'capability': 'wrong'})
        dispatch(self.host, 'POST', f'/api/view-contexts/{first["context_id"]}/release', {'capability': first['capability']})
        self.assertTrue(self.readers[0].running)
        self.read(second)
        self.now = 16
        with self.assertRaises(HostError):
            dispatch(self.host, 'POST', f'/api/view-contexts/{second["context_id"]}/heartbeat', {'capability': second['capability']})
        self.assertFalse(self.readers[0].running)

    def test_uncertain_reap_retains_slot_and_never_restarts_automatically(self):
        lease = self.acquire()
        self.readers[0].stop_result = False
        with self.assertRaises(HostError) as error:
            self.acquire(1, context_id=lease['context_id'], capability=lease['capability'])
        self.assertEqual(error.exception.code, 'cleanup_pending')
        self.host.tick()
        self.assertEqual(len(self.readers), 1)
        self.assertTrue(self.host.stopping)
        self.readers[0].stop_result = True
        self.host.tick()
        self.assertIsNone(self.host.reader)
        self.assertEqual(len(self.readers), 1)

    def test_changed_source_or_extra_interpreted_shard_cannot_activate(self):
        (self.sources[0]/'extra.safetensors').write_bytes(b'not interpreted')
        with self.assertRaises(HostError) as error:
            self.acquire()
        self.assertEqual(error.exception.code, 'fixture_unavailable')
        self.assertEqual(self.readers, [])

    def test_stale_source_after_open_disposes_owned_reader(self):
        lease = self.acquire()
        self.read(lease)
        path = self.sources[0]/'tiny.safetensors'
        path.write_bytes(path.read_bytes()[:-1]+b'x')
        with self.assertRaises(HostError) as error:
            self.read(lease, 'progress', '&tensor=0')
        self.assertEqual(error.exception.code, 'source_changed')
        self.assertFalse(self.readers[0].running)

    def test_changed_source_catalog_is_not_ready_and_has_no_lifecycle_side_effect(self):
        lease = self.acquire()
        self.read(lease)
        self.assertTrue(self.host.catalog()['models'][0]['view_ready'])
        path = self.sources[0]/'tiny.safetensors'
        path.write_bytes(path.read_bytes()[:-1]+b'x')
        with patch('atlas_host.runtime_adapter.check_fixture', side_effect=AssertionError('no payload read')):
            item = next(item for item in self.host.catalog()['models'] if item['model_id'] == self.ids[0])
        self.assertEqual(item['state'], 'source_changed')
        self.assertFalse(item['fixture_eligible'])
        self.assertFalse(item['view_ready'])
        self.assertEqual(self.readers[0].stops, 0)
        self.assertEqual(len(self.readers), 1)
        with self.assertRaises(HostError) as error:
            self.read(lease)
        self.assertEqual(error.exception.code, 'source_changed')

    def test_changed_owner_receipt_cannot_advertise_old_reader_ready(self):
        lease = self.acquire()
        self.read(lease)
        receipt = self.registry.owner_receipt(self.ids[0])
        receipt['name'] = 'Owner changed display metadata'
        with patch.object(self.registry, 'owner_receipt', return_value=receipt):
            self.assertFalse(any(item['view_ready'] for item in self.host.catalog()['models']))
        self.assertEqual(self.readers[0].stops, 0)

    def setup_failure(self, stack, outcomes):
        binary = self.root/'target/release/weight-atlas-rust'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'non-executable test placeholder; Popen is mocked')
        child = Mock()
        child.stdout.fileno.return_value = 123
        child.poll.return_value = None
        error = OSError('synthetic post-spawn pipe setup failure')
        stack.enter_context(patch('host_atlas.ROOT', self.root))
        stack.enter_context(patch('host_atlas.available', return_value=10*GIB))
        spawn = stack.enter_context(patch('host_atlas.subprocess.Popen', return_value=child))
        stack.enter_context(patch('host_atlas.os.set_blocking', side_effect=error))
        reap = stack.enter_context(patch('host_atlas.signal_and_reap', side_effect=outcomes))
        self.host.factory = lambda entry, cache: Renderer(entry, cache, 8797, self.host)
        return child, error, spawn, reap

    def test_post_spawn_setup_failure_reaps_owned_child_and_preserves_diagnostic(self):
        with ExitStack() as stack:
            child, original, spawn, reap = self.setup_failure(stack, [True])
            with self.assertRaises(HostError) as error:
                self.acquire()
            self.assertEqual(error.exception.code, 'activation_failed')
            self.assertIs(error.exception.__cause__, original)
            reap.assert_called_once_with(child, terminate=True, timeout=.2)
            child.stdout.close.assert_called_once()
            self.assertIsNone(self.host.reader)
            self.assertFalse(self.host.stopping)
            self.assertTrue(self.host.close())
            self.assertEqual(spawn.call_count, 1)

    def test_post_spawn_uncertain_reap_keeps_slot_until_confirmed(self):
        with ExitStack() as stack:
            child, original, spawn, reap = self.setup_failure(stack, [False, False, True])
            with self.assertRaises(HostError) as error:
                self.acquire()
            self.assertIs(error.exception.__cause__, original)
            self.assertIs(self.host.reader.process, child)
            self.assertTrue(self.host.stopping)
            child.stdout.close.assert_not_called()
            with self.assertRaises(HostError) as busy:
                self.acquire(1)
            self.assertEqual(busy.exception.code, 'cleanup_pending')
            self.assertEqual(spawn.call_count, 1)
            self.now += .1
            self.host.tick()
            self.assertIsNone(self.host.reader)
            self.assertFalse(self.host.stopping)
            child.stdout.close.assert_called_once()
            self.assertEqual([call.kwargs['terminate'] for call in reap.call_args_list], [True, False, False])
            # A new explicit selection is admissible only after confirmed cleanup.
            self.host.factory = lambda entry, cache: FakeRenderer(entry)
            self.acquire(1)

    def test_post_spawn_cleanup_exception_still_keeps_owned_slot(self):
        with ExitStack() as stack:
            child, original, spawn, reap = self.setup_failure(stack, [OSError('cleanup uncertain'), True])
            with self.assertRaises(HostError) as error:
                self.acquire()
            self.assertIs(error.exception.__cause__, original)
            self.assertIs(self.host.reader.process, child)
            self.assertTrue(self.host.stopping)
            child.stdout.close.assert_not_called()
            self.host.tick()
            self.assertIsNone(self.host.reader)
            child.stdout.close.assert_called_once()
            self.assertEqual(spawn.call_count, 1)

    def test_forged_renderer_correspondence_never_activates(self):
        lease = self.acquire()
        self.readers[0].info['source_directory'] = '/not-the-owner-root'
        with self.assertRaises(HostError) as error:
            self.read(lease)
        self.assertEqual(error.exception.code, 'source_changed')
        self.assertFalse(self.readers[0].running)

    def test_small_progress_and_selected_status_never_fetch_full_catalog(self):
        lease = self.acquire()
        self.read(lease)
        self.readers[0].requests.clear()
        result = self.read(lease, 'progress', '&tensor=0')
        self.read(lease, 'tensor-status', '&tensor=0')
        self.assertEqual(self.readers[0].requests, ['/api/progress?tensor=0', '/api/tensor-status?tensor=0'])
        self.assertNotIn('catalog', result)
        self.assertLessEqual(len(canonical(result)), 16384)

    def test_route_allowlist_has_no_registry_write_calibration_profile_or_download(self):
        lease = self.acquire()
        for method, path, data in [('POST', '/api/models', {}), ('POST', '/api/calibrate', {}),
                                   ('POST', '/api/profiles/start', {}), ('POST', '/api/download', {}),
                                   ('GET', '/api/owner/receipt', None)]:
            with self.subTest(path=path), self.assertRaises(HostError):
                dispatch(self.host, method, path, data)
        with self.assertRaises(ValueError):
            dispatch(self.host, 'GET', f'/api/models/{lease["model_id"]}/model?context={lease["context_id"]}&path=/tmp')

    def test_fixture_preparation_only_constructs_owned_cli_command(self):
        calls = []
        def run(command, **options):
            calls.append((command, options))
            return type('Completed', (), {'returncode': 0})()
        result = prepare({'paths': {'registry': str(self.registry.path), 'cache': str(self.root/'cache')}},
                         self.ids[0], run=run)
        self.assertTrue(result['fixture_calibrated'])
        self.assertEqual(calls[0][0][1], 'calibrate')
        self.assertIn(str(self.sources[0]), calls[0][0])
        self.assertEqual(calls[0][1], {'timeout': 5, 'check': True})

    def test_inference_routes_stay_closed_after_inference_source_integration(self):
        handler = object.__new__(HostHandler)
        handler.server = type('Server', (), {'host': self.host})()
        handler.headers = {'X-Atlas-Local': '1'}
        output = []
        handler.send = lambda status, body, mime='application/json': output.append((status, body))
        with patch('host_atlas.dispatch', side_effect=AssertionError('No inference dispatch')):
            for method, path in [('GET', '/api/inference'),
                                 ('POST', '/api/inference/start'),
                                 ('POST', '/api/inference/sweep-plan')]:
                handler.command, handler.path = method, path
                handler.validated = lambda: (path, 0)
                handler.handle_action()
                self.assertEqual(output[-1][0], 503)
                self.assertEqual(output[-1][1]['code'], 'disabled')
        self.assertEqual(self.readers, [])

    def test_actual_handler_uses_dispatch_without_constructing_http_server(self):
        handler = object.__new__(HostHandler)
        handler.server = type('Server', (), {'host': self.host})()
        handler.command, handler.path = 'GET', '/api/models'
        handler.headers = {}
        handler.validated = lambda: ('/api/models', 0)
        output = []
        handler.send = lambda status, body, mime='application/json': output.append((status, body, mime))
        handler.handle_action()
        self.assertEqual(output[0][0], 200)
        self.assertEqual(len(json.loads(output[0][1])['models']), 2)
        self.assertEqual(self.readers, [])
        handler.command, handler.path = 'POST', '/api/view-contexts'
        raw = canonical({'model_id': self.ids[0]})
        handler.validated = lambda: ('/api/view-contexts', len(raw))
        handler.rfile = io.BytesIO(raw)
        handler.handle_action()
        self.assertEqual(output[-1][0], 400, 'Missing local action header cannot activate')
        handler.headers = {'X-Atlas-Local': '1'}
        handler.rfile = io.BytesIO(raw)
        handler.handle_action()
        self.assertEqual(output[-1][0], 202)
        self.assertEqual(len(self.readers), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
