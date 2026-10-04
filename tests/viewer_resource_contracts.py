"""Pure configuration tests: no resource mutation, native work, server, or model."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import viewer_resources as budgets


class ResourceTests(unittest.TestCase):
    def test_compatibility_defaults(self):
        result = budgets.validate({})
        self.assertEqual(result['cpu_count'], 1)
        self.assertEqual(result['address_space_bytes'], 768*budgets.MIB)
        self.assertEqual(result['available_floor_bytes'], 3*budgets.GIB)
        self.assertEqual(result['disk_reserve_bytes'], 25*budgets.GIB)
        self.assertEqual(result['tile_cache_bytes'], 2*budgets.GIB)
        self.assertEqual(result['tile_cache_files'], 1000)
        result['cpu_count'] = 8
        self.assertEqual(budgets.validate({})['cpu_count'], 1)

    def test_bounded_explicit_machine_budget(self):
        selected = budgets.validate({'cpu_count': 4, 'address_space_bytes': 2*budgets.GIB,
                                     'workspace_bytes': 128*budgets.MIB,
                                     'tile_cache_bytes': 8*budgets.GIB,
                                     'tile_cache_files': 8000})
        self.assertEqual(selected['cpu_count'], 4)
        self.assertEqual(selected['available_floor_bytes'], 3*budgets.GIB)
        self.assertEqual(budgets.encode(selected), budgets.encode(dict(reversed(list(selected.items())))))

    def test_refuses_disabled_guards_and_noninteger_values(self):
        for key, (low, high) in budgets.RANGES.items():
            for value in [low-1, high+1, True, str(low), float(low), None]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    budgets.validate({key: value})
        for value in [{'bind': '0.0.0.0'}, [], None,
                      {'workspace_bytes': budgets.GIB, 'address_space_bytes': 768*budgets.MIB}]:
            with self.assertRaises(ValueError):
                budgets.validate(value)


if __name__ == '__main__':
    unittest.main()
