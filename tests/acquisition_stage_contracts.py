"""Synthetic stream and installed-publication contracts; never network or model work."""

import hashlib
import io
from pathlib import Path
import stat
from typing import Any, Callable, ParamSpec, TypeVar
import unittest
from unittest.mock import patch

import acquisition_contracts as fixtures
from atlas_host import acquisition as module

P = ParamSpec("P")
T = TypeVar("T")


class AcquisitionStageContracts(unittest.TestCase):
    def setUp(self) -> None:
        denied = patch("socket.socket", side_effect=AssertionError("No network"))
        denied.start()
        self.addCleanup(denied.stop)

    def ok(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid synthetic call raised {type(error).__name__}: {error}")

    def fixture(self) -> fixtures.Contracts:
        case = fixtures.Contracts(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        return case

    def retained(
        self, case: fixtures.Contracts, *, reason: str, publication: str, durable: bool
    ) -> dict[str, Any]:
        return {
            "installed": True,
            "registered": None if publication == "unconfirmed" else False,
            "enabled": None if publication == "unconfirmed" else False,
            "inference_ready": False,
            "plan_digest": case.reviewed["plan_digest"],
            "reason": reason,
            "installation_durability": (
                "parent_directory_fsync_confirmed" if durable else "unconfirmed"
            ),
            "registry_publication": publication,
        }

    def assert_retained_bytes(self, case: fixtures.Contracts) -> None:
        for name, raw in case.data.items():
            self.assertEqual((case.root / "installed" / name).read_bytes(), raw)
        self.assertEqual(list(case.root.glob(".atlas-acquire-*")), [])

    def test_stream_preserves_read_sizes_reservations_permissions_and_exact_bytes(
        self,
    ) -> None:
        case = self.fixture()
        case.data["model.safetensors"] = b"a" * 65537
        case.manifest["files"] = [
            {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in case.data.items()
        ]
        case.reviewed = module.plan(case.manifest, "Tiny", max_bytes=70000)
        reads: dict[str, list[int]] = {}
        streams: list[io.BytesIO] = []
        fetches: list[tuple[str, float]] = []

        def fetch(url: str, timeout: float) -> io.BytesIO:
            name = url.rsplit("/", 1)[1]
            reads[name] = []
            fetches.append((url, timeout))

            class Stream(io.BytesIO):
                def read(self, count: int = -1) -> bytes:
                    reads[name].append(count)
                    return super().read(count)

            source = Stream(case.data[name])
            streams.append(source)
            return source

        with patch.object(module, "reservation", wraps=module.reservation) as reserve:
            result = self.ok(
                case.run_acquire, max_bytes=70000, fetch=fetch, clock=lambda: 0.0
            )
        self.assertIs(result["registered"], True)
        self.assertIs(result["enabled"], False)
        self.assertEqual(
            reads,
            {
                "LICENSE": [len(case.data["LICENSE"]), 1],
                "model.safetensors": [65536, 1, 1],
            },
        )
        self.assertTrue(all(source.closed for source in streams))
        self.assertEqual(
            fetches,
            [
                (f'https://huggingface.co/fixtures/tiny/resolve/{"a" * 40}/{name}', 10)
                for name in ["LICENSE", "model.safetensors"]
            ],
        )
        total = 65537 + len(case.data["LICENSE"])
        self.assertEqual(
            [item.kwargs["remaining_download"] for item in reserve.call_args_list],
            [total, total, total, total, 65537, 65537, 65537, 65537, 1, 1, 0, 0],
        )
        self.assert_retained_bytes(case)
        for name in case.data:
            self.assertEqual(
                stat.S_IMODE((case.root / "installed" / name).stat().st_mode), 0o600
            )

    def test_success_preserves_complete_publication_receipt_and_budget(self) -> None:
        case = self.fixture()
        receipt = {"model_id": "synthetic", "enabled": False}
        with patch.object(case.registry, "register", return_value=receipt) as register:
            result = self.ok(case.run_acquire, clock=lambda: 0.0)
        self.assertEqual(
            result,
            {
                **receipt,
                "installed": True,
                "registered": True,
                "inference_ready": False,
                "plan_digest": case.reviewed["plan_digest"],
                "installation_durability": "parent_directory_fsync_confirmed",
                "registry_publication": "confirmed",
            },
        )
        register.assert_called_once_with(
            case.root / "installed",
            case.reviewed["manifest"],
            "Tiny",
            max_bytes=100,
            timeout_ms=5000,
            publish_disabled=True,
        )
        self.assertEqual(receipt, {"model_id": "synthetic", "enabled": False})
        self.assert_retained_bytes(case)

    def test_registration_failures_preserve_complete_retained_receipts(self) -> None:
        for error, reason, publication in [
            (
                OSError("synthetic IO uncertainty"),
                "registry_publication_unconfirmed",
                "unconfirmed",
            ),
            (
                ValueError("synthetic verification refusal"),
                "verified_data_retained_for_owner_registration",
                "not_published",
            ),
        ]:
            with self.subTest(error=type(error).__name__):
                case = self.fixture()
                with patch.object(
                    case.registry, "register", side_effect=error
                ) as register:
                    result = self.ok(case.run_acquire, clock=lambda: 0.0)
                self.assertEqual(
                    result,
                    self.retained(
                        case, reason=reason, publication=publication, durable=True
                    ),
                )
                self.assertEqual(register.call_count, 1)
                self.assert_retained_bytes(case)
                self.assertFalse(case.registry.path.exists())

    def test_prepublication_failures_keep_no_registration_and_uncertain_durability(
        self,
    ) -> None:
        for error, reason in [
            (
                OSError("synthetic directory uncertainty"),
                "post_install_io_error_data_retained",
            ),
            (
                ValueError("synthetic post-install refusal"),
                "post_install_admission_failed_data_retained",
            ),
        ]:
            with self.subTest(error=type(error).__name__):
                case = self.fixture()
                actual = module.os.fsync

                def sync(fd: int) -> None:
                    if stat.S_ISDIR(module.os.fstat(fd).st_mode):
                        raise error
                    actual(fd)

                with (
                    patch.object(module.os, "fsync", side_effect=sync),
                    patch.object(case.registry, "register") as register,
                ):
                    result = self.ok(case.run_acquire, clock=lambda: 0.0)
                self.assertEqual(
                    result,
                    self.retained(
                        case, reason=reason, publication="not_attempted", durable=False
                    ),
                )
                register.assert_not_called()
                self.assert_retained_bytes(case)

    def test_postinstall_deadline_keeps_complete_receipt_and_owned_data(self) -> None:
        case = self.fixture()
        now = [0.0]
        actual = module._install_new_directory

        def install(source: Path, destination: Path) -> None:
            actual(source, destination)
            now[0] = 0.2

        with (
            patch.object(module, "_install_new_directory", side_effect=install),
            patch.object(case.registry, "register") as register,
        ):
            result = self.ok(case.run_acquire, timeout_ms=100, clock=lambda: now[0])
        self.assertEqual(
            result,
            self.retained(
                case,
                reason="deadline_after_install_data_retained",
                publication="not_attempted",
                durable=True,
            ),
        )
        register.assert_not_called()
        self.assert_retained_bytes(case)


if __name__ == "__main__":
    unittest.main()
