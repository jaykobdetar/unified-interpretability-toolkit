import hashlib
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from analytics.core import analyze, digest, fold, permutation, resolve_heads, statistics
from analytics.source import Catalog, Tensor, read_layout_evidence
from analytics.svd import run as svd_run

REGION = {'row': 0, 'col': 0, 'rows': 2, 'cols': 3}


class CoreTests(unittest.TestCase):
    def report(self, values=(1, -2, 0, 4, 5, -6), **kw):
        return analyze(values, source_identity='fixture-source-v1', tensor='matrix', shape=[2, 3], region=REGION, **kw)

    def test_independent_hand_calculated_reference(self):
        result = self.report()['original']
        self.assertEqual([x['mean_abs'] for x in result['rows']], [1, 5])
        self.assertEqual([x['mean_abs'] for x in result['columns']], [2.5, 3.5, 3])
        self.assertEqual(result['row_order'], [1, 0])
        self.assertEqual(result['column_order'], [1, 2, 0])
        self.assertEqual([(x['native_indices'], x['value']) for x in result['top_values'][:3]],
                         [([1, 2], -6), ([1, 1], 5), ([1, 0], 4)])

    def test_shuffle_preserves_exact_bits_and_native_maps(self):
        values = [0.0, -0.0, 2., 2., -4., 8.]
        report = self.report(values, seed=42)
        order = report['control']['position_to_source']
        self.assertEqual(sorted(order), list(range(6)))
        self.assertEqual(order, permutation(6, 42))
        self.assertNotEqual(order, permutation(6, 43))
        bits = lambda xs: sorted(struct.pack('>d', x) for x in xs)
        self.assertEqual(bits(values), bits([values[i] for i in order]))
        for x in report['shuffled']['top_values']:
            source = order[x['index']]
            self.assertEqual(x['source_native_indices'], [source//3, source%3])
        self.assertEqual(values, [0.0, -0.0, 2., 2., -4., 8.])

    def test_sorting_manufactures_gradient_without_geometry_change(self):
        values = [0.2, -0.9, 0.1, -0.5]
        region = {'row': 0, 'col': 0, 'rows': 4, 'cols': 1}
        result = statistics(values, [4, 1], region)
        self.assertEqual(result['row_order'], [1, 3, 0, 2])
        self.assertEqual([x['index'] for x in result['rows']], list(range(4)))
        ranked = [result['rows'][i]['mean_abs'] for i in result['row_order']]
        self.assertEqual(ranked, sorted(map(abs, values), reverse=True))
        self.assertEqual(values, [0.2, -0.9, 0.1, -0.5])

    def test_zero_ties_and_finite_rejection(self):
        result = self.report([0]*6)
        self.assertEqual(result['original']['row_order'], [0, 1])
        for invalid in [math.nan, math.inf, -math.inf, 1e308, True]:
            with self.assertRaises(ValueError): self.report([invalid]*6)
        for invalid in [-1, 2**32, 1.5, True]:
            with self.assertRaises(ValueError): self.report(seed=invalid)
        with self.assertRaises(ValueError): self.report([1])
        with self.assertRaises(ValueError): self.report(top=33)

    def test_offset_coordinates_and_vector(self):
        region = {'row': 0, 'col': 5, 'rows': 1, 'cols': 3}
        result = analyze([1, -4, 2], source_identity='s', tensor='v', shape=[10], region=region)
        self.assertEqual([x['index'] for x in result['original']['vector']], [5, 6, 7])
        self.assertEqual(result['original']['top_values'][0]['native_indices'], [6])
        self.assertFalse(result['coverage']['full_tensor'])
        self.assertFalse(result['coverage']['full_model'])

    def test_caps_and_key_identity(self):
        for shape, region in [([5000], {'row': 0, 'col': 0, 'rows': 1, 'cols': 5000}),
                              ([257, 256], {'row': 0, 'col': 0, 'rows': 257, 'cols': 256}),
                              ([0], {'row': 0, 'col': 0, 'rows': 1, 'cols': 1})]:
            with self.assertRaises(ValueError): analyze([], source_identity='s', tensor='t', shape=shape, region=region)
        self.assertNotEqual(self.report(seed=1)['cache_key'], self.report(seed=2)['cache_key'])
        a = self.report()
        b = analyze([1, -2, 0, 4, 5, -6], source_identity='different', tensor='matrix', shape=[2, 3], region=REGION)
        self.assertNotEqual(a['cache_key'], b['cache_key'])


class HeadTests(unittest.TestCase):
    def setUp(self):
        self.config = {'hidden_size': 4, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'num_hidden_layers': 1}
        self.evidence = {'config_sha256': 'fixture-config', 'config_canonical_sha256': digest(self.config), 'implementation_sha256': 'fixture-layout'}
        self.profiles = {('fixture-config', 'fixture-layout'): 'separate-contiguous-linear-out-in-v1'}

    def resolve(self, projection, shape, **kw):
        return resolve_heads(f'model.layers.0.self_attn.{projection}_proj.weight', shape, self.config,
                             kw.get('evidence', self.evidence), kw.get('profiles', self.profiles))

    def test_q_kv_o_distinctions(self):
        q, k, o = self.resolve('q', [4, 4]), self.resolve('k', [2, 4]), self.resolve('o', [4, 4])
        self.assertEqual((q['axis'], q['head_count'], q['boundaries']), ('row', 2, [0, 2, 4]))
        self.assertEqual((k['axis'], k['head_count']), ('row', 1))
        self.assertEqual((o['axis'], o['head_count']), ('column', 2))
        self.assertIn('Output-projection', o['label'])

    def test_ambiguous_missing_or_wrong_layout_fails_closed(self):
        self.assertFalse(self.resolve('q', [4, 4], profiles={})['available'])
        self.assertFalse(self.resolve('q', [4, 4], evidence=None)['available'])
        self.assertFalse(self.resolve('k', [4, 2])['available'])
        self.assertFalse(self.resolve('qkv', [4, 4])['available'])
        self.config['hidden_size'] = 8  # Hash mismatch must invalidate evidence.
        self.assertFalse(self.resolve('q', [4, 8])['available'])

    def test_fold_independent_arithmetic_and_partial_counts(self):
        q = self.resolve('q', [4, 4])
        full = fold(list(range(1, 17)), {'row': 0, 'col': 0, 'rows': 4, 'cols': 4}, q)
        self.assertEqual([x['mean_abs'] for x in full['offsets']], [6.5, 10.5])
        self.assertEqual([x['count'] for x in full['offsets']], [8, 8])
        self.assertTrue(full['full_head_axis'])
        self.assertEqual(full['matrix']['mean'], [5, 6, 7, 8, 9, 10, 11, 12])
        self.assertEqual(full['matrix']['counts'], [2]*8)
        partial = fold([5, 6], {'row': 1, 'col': 0, 'rows': 1, 'cols': 2}, q)
        self.assertFalse(partial['full_head_axis'])
        self.assertIsNone(partial['offsets'][0]['mean_abs'])
        self.assertEqual(partial['offsets'][1]['mean_abs'], 5.5)
        o = self.resolve('o', [4, 4])
        full_o = fold(list(range(1, 17)), {'row': 0, 'col': 0, 'rows': 4, 'cols': 4}, o)
        self.assertEqual([x['mean_abs'] for x in full_o['offsets']], [8, 9])
        self.assertEqual(full_o['matrix']['mean'], [2, 3, 6, 7, 10, 11, 14, 15])
        cancel = fold([1]*8+[-1]*8, {'row': 0, 'col': 0, 'rows': 4, 'cols': 4}, q)
        self.assertEqual(cancel['matrix']['mean'], [0]*8)
        self.assertEqual(cancel['matrix']['mean_abs'], [1]*8)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # These are trusted fixture catalog extents, not a safetensors loader test.
        self.path = self.root/'fixture.safetensors'
        self.path.write_bytes(b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', x))[0] >> 16) for x in [1., -2., 0., 4., 5., -6., 7., 8.]))
        self.tensors = [Tensor('matrix', (2, 3), self.path.name, 0), Tensor('vector', (2,), self.path.name, 12)]
        self.catalog = Catalog(self.root, self.tensors, 'validated-fixture-v1')

    def tearDown(self): self.temp.cleanup()

    def test_bounded_source_exact_coordinates(self):
        region = {'row': 1, 'col': 1, 'rows': 1, 'cols': 2}
        result = self.catalog.region('matrix', region)
        self.assertEqual(result['original']['top_values'][0]['native_indices'], [1, 2])
        self.assertEqual(result['original']['top_values'][0]['value'], -6)
        self.assertEqual(result['coverage']['visited_values'], 2)
        self.assertIs(self.catalog.validate_cached(result), result)

    def test_model_coverage_and_partial_axis(self):
        full = self.catalog.model_outliers()
        self.assertTrue(full['coverage']['full_model'])
        self.assertEqual(full['coverage']['visited_values'], 8)
        self.assertEqual(full['rankings']['original']['values'][0]['native_indices'], [1])
        partial = self.catalog.model_outliers(value_budget=2)
        self.assertFalse(partial['coverage']['full_model'])
        self.assertEqual(partial['coverage']['visited_values'], 2)
        self.assertFalse(partial['rankings']['original']['rows'][0]['full_axis'])
        self.assertEqual(partial['coverage']['tensors'][1]['visited_values'], 0)

    def test_source_and_cache_changes_rejected(self):
        report = self.catalog.region('matrix', REGION)
        with self.assertRaises(ValueError): self.catalog.validate_cached({**report, 'source_identity': 'other'})
        self.path.write_bytes(self.path.read_bytes()[:-2])
        with self.assertRaises(ValueError): self.catalog.read(self.tensors[0], REGION)
        with self.assertRaises(ValueError): self.catalog.validate_cached(report)

    def test_path_and_extent_rejection(self):
        with self.assertRaises(ValueError): Catalog(self.root, [Tensor('x', (2,), '../escape.safetensors', 0)], 's')
        with self.assertRaises(ValueError): Catalog(self.root, [Tensor('x', (100,), self.path.name, 0)], 's')
        link = self.root/'link.safetensors'; link.symlink_to(self.path)
        with self.assertRaises(ValueError): Catalog(self.root, [Tensor('x', (1,), link.name, 0)], 's')

    def test_layout_evidence_hashes_and_duplicate_rejection(self):
        config = self.root/'config.json'; impl = self.root/'layout.py'
        config.write_text('{"hidden_size": 4}'); impl.write_text('# fixture-only reviewed layout')
        data, evidence = read_layout_evidence(config, impl)
        self.assertEqual(evidence['config_sha256'], hashlib.sha256(config.read_bytes()).hexdigest())
        self.assertEqual(evidence['config_canonical_sha256'], digest(data))
        config.write_text('{"x": 1, "x": 2}')
        with self.assertRaises(ValueError): read_layout_evidence(config, impl)

    def test_svd_large_window_excluded_without_worker(self):
        region = {'row': 0, 'col': 0, 'rows': 65, 'cols': 1}
        result = svd_run([0]*65, [65, 1], region)
        self.assertFalse(result['available'])
        self.assertIn('64 by 64', result['reason'])


if __name__ == '__main__': unittest.main()
