"""Acquisition boundary contracts use injected streams and inert OS/network doubles."""

from copy import deepcopy
import ctypes
import errno
from pathlib import Path
import stat
from types import SimpleNamespace as NS
from typing import Callable, ParamSpec, TypeVar
import unittest
from unittest.mock import Mock, patch

import acquisition_contracts as fixtures
from atlas_host import acquisition as module
from atlas_host.common import identity

P = ParamSpec("P")
T = TypeVar("T")


class AcquisitionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.socket_patch = patch(
            "socket.socket", side_effect=AssertionError("No network")
        )
        self.socket_patch.start()
        self.addCleanup(self.socket_patch.stop)

    def ok(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid inert call raised {type(error).__name__}: {error}")

    def fixture(self) -> fixtures.Contracts:
        case = fixtures.Contracts(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        return case

    def test_plan_pins_receipt_fields_and_input_copy(self) -> None:
        case = self.fixture()
        before = deepcopy(case.manifest)
        result = self.ok(module.plan, case.manifest, "Tiny", max_bytes=100)
        self.assertEqual(result["name"], "Tiny")
        self.assertEqual(result["payload_bytes"], sum(map(len, case.data.values())))
        self.assertEqual(result["metadata_bytes"], module.MAX_REGISTRY_BYTES + 65536)
        self.assertEqual(result["max_bytes"], 100)
        self.assertEqual(result["cache_growth_bytes"], 0)
        self.assertIs(result["enabled_after_acquisition"], False)
        self.assertIs(result["inference_ready"], False)
        body = {key: value for key, value in result.items() if key != "plan_digest"}
        self.assertEqual(
            result["plan_digest"], identity("weight-atlas-acquire-plan-v1", body)
        )
        self.assertEqual(case.manifest, before)
        result["manifest"]["files"][0]["bytes"] = 0
        self.assertEqual(case.manifest, before)

    def test_url_preserves_repository_revision_and_filename(self) -> None:
        case = self.fixture()
        self.assertEqual(
            self.ok(module._url, case.manifest, "LICENSE"),
            "https://huggingface.co/fixtures/tiny/resolve/" + "a" * 40 + "/LICENSE",
        )

    def test_allowed_url_reads_once_and_preserves_return_and_refusal(self) -> None:
        url = "https://huggingface.co/fixtures/tiny/resolve/" + "a" * 40 + "/LICENSE"
        with patch.object(module, "urlsplit", wraps=module.urlsplit) as split:
            self.assertIsNone(self.ok(module._allowed_url, url))
        split.assert_called_once_with(url)
        with self.assertRaises(ValueError) as caught:
            module._allowed_url("http://huggingface.co/LICENSE")
        self.assertEqual(str(caught.exception), "Unapproved acquisition redirect")

    def test_redirect_forwards_original_arguments_and_parent_result(self) -> None:
        url = "https://huggingface.co/LICENSE"
        request, stream, headers, result = object(), object(), object(), object()
        with (
            patch.object(module, "_allowed_url", wraps=module._allowed_url) as allowed,
            patch.object(
                module.HTTPRedirectHandler, "redirect_request", return_value=result
            ) as parent,
        ):
            actual = self.ok(
                module._Redirects().redirect_request,
                request,
                stream,
                302,
                "Found",
                headers,
                url,
            )
        self.assertIs(actual, result)
        allowed.assert_called_once_with(url)
        parent.assert_called_once_with(request, stream, 302, "Found", headers, url)

    def test_public_open_uses_empty_proxy_and_exact_timeout(self) -> None:
        url, result = "https://huggingface.co/LICENSE", object()
        opener = NS(open=Mock(return_value=result))
        with (
            patch.object(module, "_allowed_url", wraps=module._allowed_url) as allowed,
            patch.object(module, "build_opener", return_value=opener) as build,
        ):
            self.assertIs(self.ok(module.open_public_data, url, 1.25), result)
        allowed.assert_called_once_with(url)
        self.assertEqual(build.call_count, 1)
        proxy, redirects = build.call_args.args
        self.assertIs(type(proxy), module.ProxyHandler)
        self.assertEqual(proxy.proxies, {})
        self.assertIs(type(redirects), module._Redirects)
        opener.open.assert_called_once_with(url, timeout=1.25)

    def test_install_preserves_atomic_arguments_and_none_return(self) -> None:
        rename = Mock(return_value=0)
        with patch.object(
            module.ctypes, "CDLL", return_value=NS(renameat2=rename)
        ) as library:
            self.assertIsNone(
                self.ok(
                    module._install_new_directory,
                    Path("/inert/stage"),
                    Path("/inert/final"),
                )
            )
        library.assert_called_once_with(None, use_errno=True)
        rename.assert_called_once_with(-100, b"/inert/stage", -100, b"/inert/final", 1)
        self.assertEqual(
            rename.argtypes,
            [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ],
        )
        self.assertIs(rename.restype, ctypes.c_int)

    def test_install_error_preserves_errno_text_and_observation_count(self) -> None:
        rename = Mock(return_value=-1)
        with (
            patch.object(module.ctypes, "CDLL", return_value=NS(renameat2=rename)),
            patch.object(
                module.ctypes, "get_errno", return_value=errno.EIO
            ) as observed,
            self.assertRaises(OSError) as caught,
        ):
            module._install_new_directory(Path("/inert/stage"), Path("/inert/final"))
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(
            caught.exception.strerror, "Atomic no-replace model installation failed"
        )
        observed.assert_called_once_with()

    def test_slot_preserves_reservation_creation_lock_and_single_close(self) -> None:
        path = Mock()
        path.name = "registry.json"
        path.with_name.return_value = Path("/inert/registry.json.acquire.lock")
        registry = NS(path=path, _disk_guard=Mock())
        with (
            patch.object(module.os, "open", return_value=71) as opened,
            patch.object(
                module.os, "fstat", return_value=NS(st_mode=stat.S_IFREG | 0o600)
            ) as observed,
            patch.object(module.fcntl, "flock") as locked,
            patch.object(module.os, "close") as closed,
        ):
            with module._slot(registry) as result:
                self.assertIsNone(result)
                closed.assert_not_called()
        registry._disk_guard.assert_called_once_with(module.MAX_REGISTRY_BYTES + 65536)
        path.parent.mkdir.assert_called_once_with(
            parents=True, exist_ok=True, mode=0o700
        )
        path.with_name.assert_called_once_with("registry.json.acquire.lock")
        opened.assert_called_once_with(
            path.with_name.return_value,
            module.os.O_RDWR
            | module.os.O_CREAT
            | module.os.O_NOFOLLOW
            | module.os.O_NONBLOCK,
            0o600,
        )
        observed.assert_called_once_with(71)
        locked.assert_called_once_with(71, module.fcntl.LOCK_EX | module.fcntl.LOCK_NB)
        closed.assert_called_once_with(71)

    def test_acquire_keeps_disabled_receipt_and_exact_registration_budget(self) -> None:
        case = self.fixture()
        with patch.object(
            case.registry, "register", wraps=case.registry.register
        ) as register:
            result = self.ok(case.run_acquire, clock=lambda: 0.0)
        self.assertIs(result["installed"], True)
        self.assertIs(result["registered"], True)
        self.assertIs(result["enabled"], False)
        self.assertIs(result["inference_ready"], False)
        self.assertEqual(result["plan_digest"], case.reviewed["plan_digest"])
        self.assertEqual(
            result["installation_durability"], "parent_directory_fsync_confirmed"
        )
        self.assertEqual(result["registry_publication"], "confirmed")
        self.assertEqual(register.call_count, 1)
        self.assertEqual(
            register.call_args.kwargs,
            {"max_bytes": 100, "timeout_ms": 5000, "publish_disabled": True},
        )
        self.assertEqual(case.registry.catalog()["models"], [])
        for name, raw in case.data.items():
            self.assertEqual((case.root / "installed" / name).read_bytes(), raw)

    def test_nested_check_preserves_every_reservation_argument(self) -> None:
        case = self.fixture()
        with patch.object(module, "reservation", wraps=module.reservation) as reserve:
            self.ok(case.run_acquire, clock=lambda: 0.0)
        self.assertGreater(reserve.call_count, 1)
        self.assertEqual(
            reserve.call_args_list[0].kwargs["remaining_download"],
            sum(map(len, case.data.values())),
        )
        self.assertEqual(reserve.call_args_list[-1].kwargs["remaining_download"], 0)
        for call in reserve.call_args_list:
            self.assertEqual(call.args, (100 * 1024**3,))
            self.assertEqual(call.kwargs["cache_growth"], 0)
            self.assertEqual(call.kwargs["metadata"], module.MAX_REGISTRY_BYTES + 65536)
            self.assertEqual(
                set(call.kwargs), {"remaining_download", "cache_growth", "metadata"}
            )


if __name__ == "__main__":
    unittest.main()
