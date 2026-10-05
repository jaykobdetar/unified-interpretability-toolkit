"""Run only under granted serialized heavy slot, using existing NumPy interpreter.

python tests/analytics/svd_heavy.py
Analytic references: diag(3,4) -> singular values 4,3, energies 16/25,9/25;
outer([1,2],[1,2]) has Frobenius energy 25 and rank one.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analytics.svd import run


class SvdTests(unittest.TestCase):
    def get(self, values):
        result = run(
            values,
            [2, 2],
            {"row": 0, "col": 0, "rows": 2, "cols": 2},
            python=sys.executable,
        )
        self.assertTrue(result["available"], result)
        return result["results"]

    def test_diagonal_reference(self):
        result = self.get([3, 0, 0, 4])["original"]
        self.assertEqual(result["singular_values"], [4, 3])
        self.assertEqual(result["energy_fractions"], [16 / 25, 9 / 25])
        self.assertAlmostEqual(result["rank_one_residual_energy_fraction"], 9 / 25)
        self.assertEqual(result["rank_one_residual"], [3, 0, 0, 0])

    def test_rank_one_and_zero(self):
        result = self.get([1, 2, 2, 4])["original"]
        self.assertAlmostEqual(result["rank_one_residual_energy_fraction"], 0)
        zero = self.get([0] * 4)
        for side in zero.values():
            self.assertIsNone(side["rank_one_residual_energy_fraction"])
            self.assertEqual(side["energy_fractions"], [None, None])

    def test_controls_refit_and_energy_conservation(self):
        result = self.get([1, 2, 3, 5])
        for side in result.values():
            self.assertAlmostEqual(sum(x * x for x in side["singular_values"]), 39)
            self.assertAlmostEqual(sum(side["energy_fractions"]), 1)
            self.assertAlmostEqual(
                side["rank_one_residual_energy_fraction"],
                1 - side["energy_fractions"][0],
            )


if __name__ == "__main__":
    unittest.main()
