"""Stdlib fixture/mock host qualification. No server, ML runtime, build or network."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from atlas_host import cache, cli, config, progress, registry
from atlas_host.budget import WorkGrant
from atlas_host.common import canonical, read_json
from atlas_host.inference import configuration_evidence, head_layout_metadata

FIXTURE = ROOT/'fixtures/tiny-bf16'
MANIFEST = json.loads((FIXTURE/'host-manifest.json').read_text())
MODEL_ID = 'm_'+registry.content_digest(MANIFEST)
BINDING_V2 = json.loads((ROOT/'tests/fixtures/host-source-binding-v2.json').read_text())


def binding_value(shape=None, leading=None):
    shape = shape or [2, 3]
    return {**deepcopy(BINDING_V2), 'shape': shape,
            'rows': shape[-2] if len(shape) > 1 else 1, 'cols': shape[-1],
            'slice': {'leading_indices': leading or [],
                      'display_axes': [0] if len(shape) == 1 else [len(shape)-2, len(shape)-1]}}


def record(kind='calibration', **changes):
    value = {'model_id': MODEL_ID, 'kind': kind,
             'binding': binding_value(),
             'state': 'running', 'visited_values': 2, 'total_values': 6,
             'elapsed_active_ms': 1, 'remaining_authorized_work': {'values': 4, 'wall_ms': 4999, 'cpu_ms': 3999},
             'complete': False, 'error': None}
    value.update(changes)
    return value


class ConfigContracts(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT/'config/atlas-host.example.json').read_text())

    def test_relative_paths_are_config_relative_and_public_limits_have_no_paths(self):
        actual = config.validate_config(self.value, '/tmp/config-place')
        self.assertEqual(actual['paths']['registry'], '/tmp/cache-host/registry.json')
        self.assertNotIn('paths', config.capabilities(actual))
        self.assertFalse(config.capabilities(actual)['downloads'])
        self.assertEqual(actual['limits']['numeric_workers'], 1)
        self.assertEqual(actual['limits']['rust_address_space_bytes'], 768*1024**2)
        self.assertEqual(actual['limits']['disk_reserve_bytes'], 25*1024**3)

    def test_configuration_does_not_apply_affinity_or_limits(self):
        with patch('os.sched_setaffinity', side_effect=AssertionError('no affinity mutation')):
            config.validate_config(self.value, '/tmp')

    def test_unreviewed_profile_network_and_cap_changes_refuse(self):
        for delta in ({'profile': 'pod-v1'}, {'bind': '0.0.0.0'},
                      {'limits': {'numeric_workers': 2}}, {'limits': {'disk_reserve_bytes': 0}},
                      {'limits': {'heavy_jobs': True}}, {'unknown': 1}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                config.validate_config({**self.value, **delta}, '/tmp')

    def test_reserved_or_duplicate_ports_refuse(self):
        for ports in ({'coordinator': 8775, 'renderer': 8797},
                      {'coordinator': 8796, 'renderer': 8796}):
            with self.assertRaises(ValueError):
                config.validate_config({**self.value, 'ports': ports}, '/tmp')

    def test_json_loader_refuses_duplicate_keys_and_oversized_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.json'
            path.write_text('{"version":1,"version":2}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_json(path, 100)
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                read_json(path, 8)


class RegistryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'fixture'
        shutil.copytree(FIXTURE, self.source)
        # Pure quota simulation for a few hundred fixture bytes, not a runtime override.
        self.registry = registry.Registry(self.root/'host/registry.json', free_bytes=lambda _: 100*1024**3)

    def register(self, **kwargs):
        return self.registry.register(self.source, MANIFEST, 'Synthetic fixture', **kwargs)

    def test_disabled_by_default_then_path_free_installed_catalog(self):
        result = self.register()
        self.assertEqual(result['verified_bytes'], 244)
        self.assertFalse(result['renderer_ready'])
        self.assertEqual(self.registry.catalog()['models'], [])
        self.registry.set_enabled(result['model_id'], True)
        catalog = self.registry.catalog()
        self.assertEqual(len(catalog['models']), 1)
        self.assertNotIn(str(self.source), json.dumps(catalog))
        self.assertNotIn('tiny.safetensors', json.dumps(catalog))
        self.assertEqual(catalog['models'][0]['state'], 'verified_pending_renderer')
        self.assertFalse(catalog['models'][0]['inference_ready'])
        self.assertEqual(self.registry.owner_receipt(result['model_id'])['root'], str(self.source))

    def test_content_identity_is_portable_and_revision_sensitive(self):
        first = self.register()['model_id']
        second_root = self.root/'other-location'
        shutil.copytree(self.source, second_root)
        second = self.registry.register(second_root, MANIFEST, 'Same content')['model_id']
        self.assertEqual(first, second)
        changed = deepcopy(MANIFEST)
        changed['revision'] = 'b'*40
        self.assertNotEqual(registry.content_digest(changed), first[2:])

    def test_source_change_prevents_enable_and_is_reported_without_rehash_claim(self):
        key = self.register()['model_id']
        self.registry.set_enabled(key, True)
        path = self.source/'tiny.safetensors'
        path.write_bytes(path.read_bytes()[:-1]+b'x')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.registry.set_enabled(key, True)
        self.assertEqual(self.registry.catalog()['models'][0]['state'], 'source_changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.register()

    def test_manifest_byte_and_time_budgets_precede_publication(self):
        with self.assertRaisesRegex(ValueError, 'byte allowance'):
            self.register(max_bytes=243)
        ticks = iter((0.0, 2.0))
        self.registry.clock = lambda: next(ticks)
        with self.assertRaisesRegex(ValueError, 'time allowance'):
            self.register()
        self.assertFalse(self.registry.path.exists())

    def test_license_and_safe_data_requirements(self):
        for changed in ({'revision': 'main'}, {'repository': 'https://example.com/model'},
                        {'license': {'id': 'Apache-2.0', 'accepted': False, 'file': None}},
                        {'provenance': 'owner_expected', 'revision': 'c'*40}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                registry.validate_manifest({**MANIFEST, **changed})
        for name in ('model.py', 'model.bin', '../tiny.safetensors'):
            changed = deepcopy(MANIFEST)
            changed['files'][0]['name'] = name
            with self.assertRaises(ValueError):
                registry.validate_manifest(changed)

    def test_owner_manifest_requires_hashed_license_file(self):
        changed = deepcopy(MANIFEST)
        changed.update(provenance='owner_expected', revision='b'*40)
        changed['license']['file'] = 'LICENSE'
        with self.assertRaisesRegex(ValueError, 'License file missing'):
            registry.validate_manifest(changed)
        changed['files'].append({'name': 'LICENSE', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()})
        self.assertEqual(registry.validate_manifest(changed)['license']['file'], 'LICENSE')

    def test_symlink_source_is_not_registered(self):
        path = self.source/'tiny.safetensors'
        path.unlink()
        path.symlink_to(FIXTURE/'tiny.safetensors')
        with self.assertRaises(OSError):
            self.register()
        self.assertFalse(self.registry.path.exists())

    def test_registry_cannot_write_inside_model(self):
        inside = registry.Registry(self.source/'registry.json', free_bytes=lambda _: 100*1024**3)
        with self.assertRaisesRegex(ValueError, 'outside model'):
            inside.register(self.source, MANIFEST, 'fixture')
        self.assertFalse(inside.path.exists())

    def test_atomic_replace_failure_preserves_previous_receipt(self):
        key = self.register()['model_id']
        before = self.registry.path.read_bytes()
        with patch.object(registry.os, 'replace', side_effect=OSError('simulated rename failure')):
            with self.assertRaises(OSError):
                self.registry.set_enabled(key, True)
        self.assertEqual(self.registry.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.registry.path.parent.iterdir()),
                         ['registry.json', 'registry.json.lock'])

    def test_writer_lock_refuses_overlap_without_losing_model(self):
        with self.registry._writer():
            with self.assertRaisesRegex(ValueError, 'writer busy'):
                self.register()
        self.assertFalse(self.registry.path.exists())

    def test_disk_reservation_counts_all_peak_components(self):
        reserve = 25*1024**3
        result = registry.reservation(reserve+1000, remaining_download=300,
                                      staging=300, cache_growth=300, metadata=100)
        self.assertEqual(result['remaining_free_bytes'], reserve)
        with self.assertRaisesRegex(ValueError, 'Insufficient disk'):
            registry.reservation(reserve+999, remaining_download=300,
                                 staging=300, cache_growth=300, metadata=100)
        self.registry._free_bytes = lambda _: reserve
        with self.assertRaisesRegex(ValueError, 'Insufficient disk'):
            self.register()
        self.assertFalse(self.registry.path.parent.exists())


class ProgressContracts(unittest.TestCase):
    def setUp(self):
        self.store = progress.ProgressStore(MODEL_ID)

    def test_private_profile_is_never_in_public_model_progress(self):
        key, token = 'a'*32, '0123456789abcdef'*4
        with self.assertRaisesRegex(ValueError, 'owner-private'):
            self.store.publish(key, record('profile'))
        self.store.publish(key, record('profile'), owner=token)
        public = self.store.snapshot()
        self.assertEqual(public['records'], [])
        self.assertNotIn(key, json.dumps(public))
        with self.assertRaisesRegex(ValueError, 'not owned'):
            self.store.snapshot(job=key, capability='wrong')
        owned = self.store.snapshot(public['cursor'], job=key, capability=token)
        self.assertTrue(owned['reset'])
        self.assertEqual(owned['records'][0]['id'], key)
        self.assertNotIn(token, json.dumps(owned))

    def test_delta_pages_never_skip_unreturned_records(self):
        for i in range(40):
            self.store.publish(f'{i:032x}', record())
        collected, cursor = [], None
        while True:
            response = self.store.snapshot(cursor)
            self.assertLessEqual(len(canonical(response)), 16384)
            self.assertLessEqual(len(response['records']), 16)
            collected.extend(item['id'] for item in response['records'])
            cursor = response['cursor']
            if not response['has_more']:
                break
        self.assertEqual(collected, [f'{i:032x}' for i in range(40)])
        self.assertEqual(self.store.snapshot(cursor)['records'], [])

    def test_restart_and_evicted_history_require_reset(self):
        self.store.publish('0'*32, record())
        cursor = self.store.snapshot()['cursor']
        for i in range(1, 70):
            self.store.publish(f'{i:032x}', record())
        self.assertEqual(len(self.store._records), 64)
        self.assertTrue(self.store.snapshot(cursor)['reset'])
        other = progress.ProgressStore(MODEL_ID)
        self.assertTrue(other.snapshot(cursor)['reset'])

    def test_snapshot_and_publish_do_not_alias_coordinator_state(self):
        source = record()
        self.store.publish('0'*32, source)
        source['binding']['shape'][0] = 99
        response = self.store.snapshot()
        response['records'][0]['binding']['shape'][0] = 88
        self.assertEqual(self.store.snapshot()['records'][0]['binding']['shape'], [2, 3])

    def test_active_private_job_survives_public_record_eviction(self):
        private, token = 'e'*32, '0123456789abcdef'*4
        self.store.publish(private, record('profile'), owner=token)
        for i in range(70):
            self.store.publish(f'{i:032x}', record())
        owned = self.store.snapshot(job=private, capability=token)
        self.assertEqual(owned['records'][0]['id'], private)

    def test_progress_cannot_mix_models_change_bindings_or_claim_false_completion(self):
        self.store.publish('0'*32, record())
        for value in (record(model_id='m_'+'c'*64), record(visited_values=1),
                      record(state='complete', complete=True), record(error='/private/source/path')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.publish('0'*32, value)
        changed = record()
        changed['binding']['source_identity'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'binding cannot change'):
            self.store.publish('0'*32, changed)

    def test_full_tensor_calibration_is_not_slice_progress(self):
        source = record('profile', total_values=12)
        source['binding'] = binding_value([2, 3, 4], [1])
        source['remaining_authorized_work']['values'] = 10
        progress.validate_progress(source)
        source['binding']['slice']['leading_indices'] = []
        with self.assertRaisesRegex(ValueError, 'leading slice'):
            progress.validate_progress(source)
        calibration = record(total_values=24)
        calibration['binding'] = binding_value([2, 3, 4], [1])
        progress.validate_progress(calibration)


class CacheContracts(unittest.TestCase):
    def setUp(self):
        self.value = {'content_digest': MODEL_ID[2:],
                      'binding': deepcopy(BINDING_V2),
                      'algorithm': 'renderer-v1', 'calibration_digest': 'c'*64,
                      'rule': 'tensor_magnitude_asinh', 'parameters': {'s': 0.25, 'D': 1.0},
                      'level': 1, 'x': 0, 'y': 0, 'control': None, 'encoding': 'png-v1'}

    def test_semantically_different_artifacts_do_not_share_identity(self):
        original = cache.derivation_identity(self.value)
        changed_values = []
        for key, changed in [('content_digest', 'd'*64), ('algorithm', 'renderer-v2'),
                             ('calibration_digest', 'e'*64), ('rule', 'tensor_magnitude'),
                             ('level', 2), ('encoding', 'png-v2')]:
            changed_values.append({**self.value, key: changed})
        changed = deepcopy(self.value)
        changed['binding']['slice']['leading_indices'] = [0]
        changed_values.append(changed)
        changed = deepcopy(self.value)
        changed['parameters']['s'] = 0.5
        changed_values.append(changed)
        changed_values.append({**self.value, 'control': {'algorithm': 'swap-or-not-8-v1',
                               'seed': 42, 'permutation_digest': 'f'*64}})
        for changed in changed_values:
            with self.subTest(changed=changed):
                self.assertNotEqual(original, cache.derivation_identity(changed))

    def test_unbound_slice_nonfinite_or_unknown_descriptor_refuses(self):
        changed = deepcopy(self.value)
        changed['binding']['slice']['leading_indices'] = []
        for value in (changed, {**self.value, 'unexpected': 1},
                      {**self.value, 'parameters': {'s': float('nan')}}):
            with self.assertRaises(ValueError):
                cache.derivation_identity(value)

    def test_frozen_sibling_binding_and_portable_derivation(self):
        self.assertEqual(cache.binding(BINDING_V2), BINDING_V2)
        original = cache.derivation_identity(self.value)
        relocated = deepcopy(self.value)
        relocated['binding']['source_identity'] = 'e'*64
        relocated['binding']['model_identity'] = 'f'*64
        self.assertEqual(cache.derivation_identity(relocated), original)
        for shape in ([4], [3, 4]):
            cache.binding(binding_value(shape))
        transposed = deepcopy(BINDING_V2)
        transposed['slice']['display_axes'] = [2, 1]
        with self.assertRaisesRegex(ValueError, 'trailing axes'):
            cache.binding(transposed)

    def test_immutable_headers_require_matching_bounded_artifact(self):
        body = b'fixture encoded image'
        digest = hashlib.sha256(body).hexdigest()
        headers = cache.immutable_headers(digest, body, 'image/png', 'tile')
        self.assertEqual(headers['ETag'], '"'+digest+'"')
        self.assertTrue(headers['Cache-Control'].startswith('private,'))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            cache.immutable_headers(digest, b'different', 'image/png', 'tile')
        for kind in ('inference', 'profile', 'partial', 'job'):
            with self.assertRaisesRegex(ValueError, 'cannot be immutable'):
                cache.immutable_headers(digest, body, 'image/png', kind)
        self.assertEqual(cache.no_store_headers(), {'Cache-Control': 'no-store'})


class BudgetContracts(unittest.TestCase):
    def setUp(self):
        self.wall, self.cpu = 0.0, 0.0
        self.grant = WorkGrant(100, 5000, 4000, clock=lambda: self.wall, cpu_clock=lambda: self.cpu)

    def test_waiting_consumes_total_wall_before_next_chunk(self):
        self.assertEqual(self.grant.advance(20, lambda n, deadline: n), 20)
        self.wall = 5.1
        with self.assertRaisesRegex(ValueError, 'Total profile budget'):
            self.grant.advance(20, lambda *_: self.fail('must not advance'))
        self.assertEqual(self.grant.visited, 20)

    def test_chunks_share_cpu_and_value_budget(self):
        self.grant.advance(80, lambda n, deadline: n)
        self.cpu = 3.0
        self.assertEqual(self.grant.advance(80, lambda n, deadline: n), 20)
        with self.assertRaisesRegex(ValueError, 'No authorized values'):
            self.grant.advance(1, lambda *_: self.fail('must not advance'))
        self.cpu = 4.1
        with self.assertRaisesRegex(ValueError, 'Total profile budget'):
            self.grant.remaining()

    def test_expired_lease_and_uncertain_chunk_cannot_restart(self):
        grant = WorkGrant(100, 5000, 4000, clock=lambda: self.wall,
                          cpu_clock=lambda: self.cpu, lease_ms=1000)
        self.wall = 1.1
        with self.assertRaisesRegex(ValueError, 'lease expired'):
            grant.heartbeat()
        with self.assertRaises(RuntimeError):
            self.grant.advance(1, lambda *_: (_ for _ in ()).throw(RuntimeError('fake chunk')))
        with self.assertRaisesRegex(ValueError, 'Grant failed'):
            self.grant.advance(1, lambda *_: self.fail('must not advance'))

    def test_chunk_overrun_and_attempted_caps_refuse(self):
        def too_slow(n, deadline):
            self.assertEqual(deadline, 5.0)
            self.wall = 6.0
            return n
        with self.assertRaisesRegex(ValueError, 'Total profile budget'):
            self.grant.advance(10, too_slow)
        self.assertTrue(self.grant.failed)
        with self.assertRaises(ValueError):
            WorkGrant(100, 6000, 4000)
        with self.assertRaisesRegex(ValueError, 'CPU clock'):
            WorkGrant(100, 5000, 4000)

    def test_cancellation_after_uncertain_work_poisoned_without_budget_reset(self):
        class SyntheticCancellation(BaseException):
            pass
        for error in (KeyboardInterrupt('synthetic interrupt'), SystemExit('synthetic exit'),
                      SyntheticCancellation('synthetic cancellation')):
            with self.subTest(error=type(error).__name__):
                grant = WorkGrant(2, 5000, 4000, clock=lambda: self.wall,
                                  cpu_clock=lambda: self.cpu)
                original = (grant.started, grant.cpu_started, grant.deadline,
                            grant.lease_end, grant.maximum, grant.cpu_ms)
                worked = []
                def interrupted(count, deadline):
                    worked.extend(range(count))
                    raise error
                with self.assertRaises(type(error)) as caught:
                    grant.advance(2, interrupted)
                self.assertIs(caught.exception, error)
                self.assertTrue(grant.failed)
                self.assertFalse(grant.in_flight)
                self.assertEqual(grant.visited, 0, 'Uncertain work is not claimed as committed progress')
                self.assertEqual(original, (grant.started, grant.cpu_started, grant.deadline,
                                           grant.lease_end, grant.maximum, grant.cpu_ms))
                with self.assertRaisesRegex(ValueError, 'Grant failed'):
                    grant.advance(2, lambda *_: self.fail('must not reauthorize uncertain values'))
                self.assertEqual(worked, [0, 1])


class InferenceBoundary(unittest.TestCase):
    def inputs(self):
        manifest = deepcopy(MANIFEST)
        manifest['files'] = [{'name': 'model.safetensors', 'bytes': 244, 'sha256': 'a'*64},
                             {'name': 'config.json', 'bytes': 10, 'sha256': 'b'*64}]
        descriptor = {'schema': 'weight-atlas-head-layout-v1', 'adapter_id': 'builtin-llama-eager',
                      'adapter_version': 1, 'evidence': 'pinned_configuration', 'runtime_verified': False,
                      'source_model': {'repo': manifest['repository'], 'revision': manifest['revision'],
                                       'weights_sha256': 'a'*64, 'config_sha256': 'b'*64}}
        return manifest, descriptor

    def test_configuration_correspondence_never_grants_runtime_fit_or_emits_worker_request(self):
        manifest, descriptor = self.inputs()
        evidence = configuration_evidence(manifest, descriptor, binding_value([3, 4]))
        self.assertTrue(evidence['source_correspondence'])
        self.assertFalse(evidence['runtime_verified'])
        self.assertFalse(evidence['fit_verified'])
        self.assertFalse(evidence['inference_ready'])
        self.assertNotIn('source_model', evidence)
        self.assertNotIn('request', evidence)

    def test_runtime_promotion_receipt_mismatch_and_higher_rank_are_rejected(self):
        manifest, descriptor = self.inputs()
        for changes in ({'runtime_verified': True, 'evidence': 'loaded_builtin_layout'},
                        {'source_model': {**descriptor['source_model'], 'config_sha256': 'c'*64}}):
            with self.assertRaises(ValueError):
                configuration_evidence(manifest, {**descriptor, **changes}, binding_value([3, 4]))
        with self.assertRaisesRegex(ValueError, 'rank-two'):
            configuration_evidence(manifest, descriptor, binding_value([2, 3, 4], [0]))

    def test_frozen_head_descriptor_requires_matching_receipt_native_projection_and_ids(self):
        descriptor = json.loads((ROOT/'tests/fixtures/host-head-layout-v1.json').read_text())
        source = descriptor['source_model']
        manifest = {**deepcopy(MANIFEST), 'repository': source['repo'], 'revision': source['revision'],
                    'files': [{'name': 'model.safetensors', 'bytes': 1, 'sha256': source['weights_sha256']},
                              {'name': 'config.json', 'bytes': 1, 'sha256': source['config_sha256']}]}
        native = binding_value([576, 576])
        native['name'] = 'model.layers.0.self_attn.o_proj.weight'
        metadata = head_layout_metadata(manifest, descriptor, native)
        self.assertEqual(metadata['head_layout_binding'], {
            'source_identity': native['source_identity'], 'model_identity': native['model_identity'],
            'weights_sha256': source['weights_sha256'], 'config_sha256': source['config_sha256']})
        self.assertFalse(metadata['head_layout']['runtime_verified'])
        self.assertFalse(metadata['host_inference_evidence']['inference_ready'])
        self.assertEqual(len(metadata['head_layout_binding']), 4)
        for wrong in ({**native, 'name': 'model.layers.30.self_attn.o_proj.weight'},
                      {**native, 'name': 'model.layers.0.self_attn.k_proj.weight'},
                      {**binding_value([192, 576]), 'name': native['name']}):
            with self.assertRaises(ValueError):
                head_layout_metadata(manifest, descriptor, wrong)
        with self.assertRaisesRegex(ValueError, 'do not match'):
            head_layout_metadata(MANIFEST, descriptor, native)
        with self.assertRaises(ValueError):
            head_layout_metadata(manifest, descriptor, None)
        with self.assertRaises(ValueError):
            head_layout_metadata(manifest, None, native)


class CLIContracts(unittest.TestCase):
    def test_fixture_plan_and_disabled_download_have_no_model_or_network_reads(self):
        args = ['--config', str(ROOT/'config/atlas-host.example.json')]
        with patch.object(registry.Registry, 'register', side_effect=AssertionError('no file hashing')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(args+['plan', '--manifest', str(FIXTURE/'host-manifest.json')]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['bytes_to_verify'], 244)
            self.assertFalse(plan['download_enabled'])
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main(args+['download']), 2)
            self.assertIn('Downloads are disabled', error.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
