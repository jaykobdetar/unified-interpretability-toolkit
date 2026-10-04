"""Lightweight direct helper/control validation; never runs an HTTP/model worker."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import inference_observations as obs
import live_inference as live


def attention():
    return {'index': 0, 'layer': 7, 'activation_site': 'attention', 'position': 2, 'input_token_id': 3,
            'attention': {'layer': 7, 'query_head': 8, 'kv_head': 2, 'head_dim': 64, 'query_position': 2,
                          'key_positions': [0, 1, 2], 'key_token_ids': [1, 2, 3],
                          'probabilities': [.25, .25, .5], 'semantics': obs.ATTENTION_SEMANTICS}}


def lens():
    return {'index': 0, 'layer': 29, 'activation_site': 'block', 'position': 2, 'token_id': 4,
            'logit_lens': {'layer': 29, 'position': 2, 'score_kind': 'raw FP32 logits', 'lens_argmax_id': 3,
                          'final_argmax_id': 4, 'semantics': obs.LENS_SEMANTICS,
                          'candidates': [{'id': 3, 'piece': 'a', 'lens_logit': 2., 'final_logit': 1., 'delta_lens_minus_final': 1.},
                                         {'id': 4, 'piece': 'b', 'lens_logit': 1., 'final_logit': 2., 'delta_lens_minus_final': -1.}]}}


class ReachedPins(Exception): pass


class Contracts(unittest.TestCase):
    def test_request_schema_and_pinned_validation_order(self):
        for layer in (0, 7, 29):
            for site, selection in [('attention', {'kind': 'attention', 'head': h}) for h in range(9)] + [('block', {'kind': 'logit_lens'})]:
                session = live.Session('unused', Path('/unused'))
                with patch.object(live, 'available', return_value=6*live.GIB), patch.object(live, 'verify_model', side_effect=ReachedPins), self.assertRaises(ReachedPins):
                    session.start({'prompt': 'public fixture', 'layer': layer, 'activation_site': site, 'observation': selection})
                self.assertIsNone(session.process)

    def test_bad_requests_stop_before_file_access(self):
        cases = [(site, value) for site in ('block', 'attention', 'mlp') for value in (None, [], True, {}, {'kind': 'other'})]
        cases += [('attention', {'kind': 'attention', 'head': h}) for h in (-1, 9, True, False, 0., '0', None)]
        cases += [('attention', {'kind': 'attention', 'head': 0, 'layers': [0]}), ('block', {'kind': 'attention', 'head': 0}),
                  ('attention', {'kind': 'logit_lens'}), ('block', {'kind': 'logit_lens', 'head': 0})]
        for site, value in cases:
            with self.subTest(site=site, value=value), patch.object(live, 'verify_model', side_effect=AssertionError('No file access')), self.assertRaises(ValueError):
                live.Session('unused', Path('/unused')).start({'prompt': 'fixture', 'activation_site': site, 'observation': value})

    def test_attention_address_distribution_and_bounds(self):
        obs.validate_record(attention(), {'kind': 'attention', 'head': 8}, 7)
        for key, value in {'query_head': 7, 'kv_head': 1, 'head_dim': 128, 'query_position': 3, 'layer': True,
                           'key_positions': [False, 1, 2], 'key_token_ids': [1, 2, True], 'probabilities': [.2, .2, .2],
                           'semantics': 'raw scores'}.items():
            step = attention();step['attention'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'attention', 'head': 8}, 7)
        for probabilities in ([True, 0., 0.], [float('nan'), 0., 1.], [float('inf'), 0., 1.], [-.1, .1, 1.], [1.], [0.]*161):
            step = attention();step['attention']['probabilities'] = probabilities
            with self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'attention', 'head': 8}, 7)
        step = attention();step['position'] = 159
        with self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'attention', 'head': 8}, 7)

    def test_only_accepted_mode_layer_site_branch(self):
        for step, selected, layer in [(attention(), None, 7), (attention(), {'kind': 'logit_lens'}, 7),
                                      (attention(), {'kind': 'attention', 'head': 8}, 8), ({}, {'kind': 'logit_lens'}, 0)]:
            with self.assertRaises(ValueError): obs.validate_record(step, selected, layer)
        step = attention();step.update(baseline={}, edited={}, activation_branch='baseline')
        with self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'attention', 'head': 8}, 7)
        step['activation_branch'] = 'edited';obs.validate_record(step, {'kind': 'attention', 'head': 8}, 7)
        step['edited'] = None;step['activation_branch'] = 'baseline';obs.validate_record(step, {'kind': 'attention', 'head': 8}, 7)

    def test_lens_union_both_values_delta_and_semantics(self):
        obs.validate_record(lens(), {'kind': 'logit_lens'}, 29)
        for key, value in {'delta_lens_minus_final': -.5, 'lens_logit': True, 'final_logit': float('nan'), 'id': True, 'piece': 'x'*1025}.items():
            step = lens();step['logit_lens']['candidates'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'logit_lens'}, 29)
        for key, value in {'candidates': [], 'lens_argmax_id': 12, 'final_argmax_id': 1, 'score_kind': 'probability', 'layer': 28}.items():
            step = lens();step['logit_lens'][key] = value
            with self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'logit_lens'}, 29)
        step = lens();step['logit_lens']['candidates'] *= 6
        with self.assertRaises(ValueError): obs.validate_record(step, {'kind': 'logit_lens'}, 29)

    def test_advertised_observations_are_bounded_and_stateless(self):
        session = live.Session('unused', Path('/unused'));session.id = 'private';session.details = {'prompt': 'private'}
        metadata = session.metadata()
        self.assertEqual(metadata['observations'], obs.schema())
        self.assertEqual(metadata['limits']['trace_steps'] if 'trace_steps' in metadata['limits'] else metadata['limits']['new_tokens'], 32)
        self.assertNotIn('private', str(metadata))
        self.assertIsNone(session.observation)


if __name__ == '__main__': unittest.main()
