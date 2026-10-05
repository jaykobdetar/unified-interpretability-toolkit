"""Pure acceptance snapshot and independent native geometry checks; no worker."""
import json
from pathlib import Path
import unittest
import inference_sweep as sweep

ROOT = Path(__file__).resolve().parents[1]


class AcceptanceFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT/'tests/fixtures/sweep-maximum-acceptance.json').read_text())
        self.plan = self.fixture['resolved_plan']

    def test_exact_current_plan_snapshot(self):
        self.assertEqual(sweep.build_plan(self.fixture['request_without_digest'], False), self.plan)

    def test_published_subset_and_output_query_geometry(self):
        plan = self.plan
        self.assertEqual(plan['version'], 'weight-atlas-sweep-plan-v2')
        self.assertEqual(plan['control_version'], 'weight-atlas-sweep-control-v2')
        self.assertEqual((plan['scope'], plan['records'], plan['prefills']), ('subset', 10, 20))
        self.assertEqual(plan['limits']['subset_limits'], {'targets':2, 'records':10, 'prefills':20})
        self.assertEqual((plan['limits']['records'], plan['limits']['prefills']), (19, 38))
        self.assertEqual((plan['limits']['wall_seconds'], plan['limits']['worker_cpu_seconds']), (120, 90))
        architecture = json.loads((ROOT/'tests/fixtures/inference-architecture.json').read_text())
        self.assertEqual(plan['architecture'], architecture)
        self.assertEqual(plan['limits']['architecture'], architecture)
        self.assertEqual(plan['digest'], '5117d5cb00e80582748b62cb58178b9a1e7833a2c96def9fd7cd978ca4994814')
        cases = plan['cases']
        self.assertEqual([c['role'] for c in cases], ['empty_control','target','matched_control','target','matched_control'])
        self.assertEqual(cases[0]['edits'], [])
        for case, start in [(cases[1], 0), (cases[2], 448)]:
            self.assertEqual(case['selected_cells'], 64*576)
            self.assertEqual(case['edits'], [{'tensor':'model.layers.0.self_attn.o_proj.weight','shape':[576,576],
                'kind':'columns','operation':'scale','start':start,'end':start+64,'scale':0.5}])
        for case, offset in [(cases[3], 3), (cases[4], 24)]:
            self.assertEqual(case['selected_cells'], 8*576)
            self.assertEqual(case['edits'], [{'tensor':'model.layers.7.self_attn.q_proj.weight','shape':[576,576],
                'kind':'rows','operation':'scale','start':head*64+offset,'end':head*64+offset+1,'scale':0.5}
                for head in range(8)])

if __name__ == '__main__':
    unittest.main()
