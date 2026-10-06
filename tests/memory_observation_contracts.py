"""Exact Linux memory observations with inert file reads; no resource changes."""

from collections.abc import Callable
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryObservation(unittest.TestCase):
    def readers(self) -> list[tuple[str, Callable[[], int]]]:
        return [
            (
                "live",
                cast(
                    Callable[[], int],
                    importlib.import_module("live_inference").available,
                ),
            ),
            (
                "profile",
                cast(
                    Callable[[], int],
                    importlib.import_module("atlas_host.profile_os").available_bytes,
                ),
            ),
            (
                "guard",
                cast(
                    Callable[[], int],
                    load(
                        "memory_fixture_core_guard", "tools/guarded-core-ui.py"
                    ).available,
                ),
            ),
        ]

    def test_first_exact_available_field_and_kib_conversion(self) -> None:
        raw = "MemFree: 43 kB\nMemAvailable: 12345 kB\nMemAvailable: 67890 kB\n"
        for name, read in self.readers():
            with (
                self.subTest(reader=name),
                patch.object(Path, "read_text", return_value=raw) as file_read,
            ):
                self.assertEqual(
                    read(), 12345 * 1024, "first MemAvailable value in bytes"
                )
                file_read.assert_called_once_with()

    def test_missing_field_and_malformed_value_keep_exceptions(self) -> None:
        for raw, error in [
            ("MemFree: 43 kB\n", StopIteration),
            ("MemAvailable: invalid kB\n", ValueError),
            ("MemAvailable:\n", IndexError),
        ]:
            for name, read in self.readers():
                with (
                    self.subTest(reader=name, raw=raw),
                    patch.object(Path, "read_text", return_value=raw),
                ):
                    with self.assertRaises(error):
                        read()

    def test_original_file_error_propagates(self) -> None:
        original = PermissionError("fixture memory read refused")
        for name, read in self.readers():
            with (
                self.subTest(reader=name),
                patch.object(Path, "read_text", side_effect=original),
            ):
                with self.assertRaises(PermissionError) as caught:
                    read()
                self.assertIs(caught.exception, original)

    def test_hash_admission_exact_floor_and_wording(self) -> None:
        guard = cast(
            Callable[[], None],
            load("memory_fixture_hash_guard", "tools/check_model_hashes.py").guard,
        )
        floor_kib = 3 * 1024**2
        for available, expected in [
            (floor_kib, (None, "")),
            (floor_kib - 1, (RuntimeError, "Paused: fewer than 3 GiB available RAM")),
        ]:
            with (
                self.subTest(available=available),
                patch.object(
                    Path, "read_text", return_value=f"MemAvailable: {available} kB\n"
                ),
            ):
                outcome: tuple[type[Exception] | None, str]
                try:
                    guard()
                    outcome = (None, "")
                except Exception as error:
                    outcome = (type(error), str(error))
                self.assertEqual(
                    outcome, expected, "hash admission floor and error text"
                )


if __name__ == "__main__":
    unittest.main()
