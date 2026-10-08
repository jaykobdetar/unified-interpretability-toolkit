"""Independent standalone-viewer defaults; no process or resource operations."""

import unittest

import viewer_resources


class ViewerPolicyVectors(unittest.TestCase):
    def test_complete_defaults_and_canonical_encoding(self) -> None:
        expected = {
            "version": 1,
            "cpu_count": 1,
            "address_space_bytes": 805306368,
            "available_floor_bytes": 3221225472,
            "disk_reserve_bytes": 26843545600,
            "workspace_bytes": 67108864,
            "tile_cache_bytes": 2147483648,
            "tile_cache_files": 1000,
        }
        self.assertEqual(viewer_resources.DEFAULTS, expected)
        self.assertEqual(list(viewer_resources.DEFAULTS), list(expected))
        actual = viewer_resources.validate({})
        self.assertEqual(actual, expected)
        self.assertTrue(all(type(value) is int for value in actual.values()))
        self.assertIsNot(actual, viewer_resources.DEFAULTS)
        self.assertEqual(
            viewer_resources.encode({}),
            '{"address_space_bytes":805306368,"available_floor_bytes":3221225472,'
            '"cpu_count":1,"disk_reserve_bytes":26843545600,"tile_cache_bytes":2147483648,'
            '"tile_cache_files":1000,"version":1,"workspace_bytes":67108864}',
        )


if __name__ == "__main__":
    unittest.main()
