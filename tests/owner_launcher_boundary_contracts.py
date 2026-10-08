"""Owner launchers are observed through inert server, signal and application doubles."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import signal
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, ParamSpec, TypeVar
import unittest
from unittest.mock import Mock, call, patch

import profile_atlas
import static_atlas

P = ParamSpec("P")
T = TypeVar("T")


class OwnerLauncherBoundaryTests(unittest.TestCase):
    def ok(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(
                f"Valid inert launcher call raised {type(error).__name__}: {error}"
            )

    def config(self) -> dict[str, Any]:
        return {
            "bind": "127.0.0.1",
            "ports": {"coordinator": 8798},
            "paths": {"registry": "registry.json", "cache": "cache"},
        }

    def exercise(self, module: ModuleType) -> tuple[Mock, Mock, str]:
        app = Mock(name="application")
        app.lifetime = SimpleNamespace(stopping=False)
        app.close.side_effect = [False, True]
        server = Mock(server_port=8798)
        server.handle_request.side_effect = lambda: setattr(
            app.lifetime, "stopping", True
        )
        output = io.StringIO()
        registry = object()
        with (
            patch.object(module, "Registry", return_value=registry) as registry_type,
            patch.object(module, "HostedApplication", return_value=app) as create,
            patch.object(module, "HTTPServer", return_value=server) as server_type,
            patch.object(module.signal, "signal") as signals,
            patch.object(module.time, "sleep") as sleep,
            redirect_stdout(output),
        ):
            self.assertIsNone(
                self.ok(module.run, self.config(), Path("binary"), "a" * 64)
            )
        registry_type.assert_called_once_with("registry.json")
        kwargs = {"dense_policy": None} if module is static_atlas else {}
        create.assert_called_once_with(
            registry, "cache", Path("binary"), "a" * 64, **kwargs
        )
        server_type.assert_called_once_with(("127.0.0.1", 8798), module.HostedHandler)
        self.assertIs(server.application, app)
        self.assertIs(server.host, app.host)
        self.assertIs(server.profile_controls, module is profile_atlas)
        self.assertEqual(server.timeout, 0.1)
        self.assertEqual(
            app.mock_calls,
            [call.start_threads(), call.tick(), call.close(), call.close()],
        )
        self.assertEqual(
            server.mock_calls, [call.handle_request(), call.server_close()]
        )
        sleep.assert_called_once_with(0.05)
        self.assertEqual(
            [c.args[0] for c in signals.call_args_list], [signal.SIGTERM, signal.SIGINT]
        )
        self.assertIs(
            signals.call_args_list[0].args[1], signals.call_args_list[1].args[1]
        )
        return signals, server, output.getvalue()

    def test_run_keeps_owned_loop_cleanup_and_announcement(self) -> None:
        for module, label in [
            (profile_atlas, "Local fixture profiles"),
            (static_atlas, "Local static host"),
        ]:
            with self.subTest(module=module.__name__):
                _, _, output = self.exercise(module)
                self.assertEqual(output, f"{label}: http://127.0.0.1:8798\n")

    def test_shutdown_callbacks_keep_exact_exception(self) -> None:
        for module in [profile_atlas, static_atlas]:
            with self.subTest(module=module.__name__):
                signals, _, _ = self.exercise(module)
                for item in signals.call_args_list:
                    try:
                        item.args[1](item.args[0], None)
                    except BaseException as error:
                        self.assertIs(type(error), KeyboardInterrupt)
                        self.assertEqual(str(error), "")
                        self.assertIsNone(error.__cause__)
                    else:
                        self.fail("Shutdown must interrupt the owned loop")

    def test_main_keeps_defaults_forwarding_and_return(self) -> None:
        for module in [profile_atlas, static_atlas]:
            with self.subTest(module=module.__name__):
                config = self.config()
                with (
                    patch.object(module, "load_config", return_value=config) as load,
                    patch.object(module, "run") as run,
                ):
                    result = self.ok(
                        module.main,
                        ["--config", "owner.json", "--binary-sha256", "a" * 64],
                    )
                self.assertIsNone(result)
                load.assert_called_once_with(Path("owner.json"))
                kwargs = (
                    {"policy_path": None, "approved_sha": None}
                    if module is static_atlas
                    else {}
                )
                run.assert_called_once_with(
                    config,
                    module.ROOT / "target/release/weight-atlas-rust",
                    "a" * 64,
                    **kwargs,
                )

    def test_static_main_keeps_explicit_owner_policy_arguments(self) -> None:
        config = self.config()
        with (
            patch.object(static_atlas, "load_config", return_value=config),
            patch.object(static_atlas, "run") as run,
        ):
            result = self.ok(
                static_atlas.main,
                [
                    "--config",
                    "owner.json",
                    "--binary",
                    "binary",
                    "--binary-sha256",
                    "a" * 64,
                    "--dense-static-policy",
                    "policy.json",
                    "--approved-dense-static-sha256",
                    "b" * 64,
                ],
            )
        self.assertIsNone(result)
        run.assert_called_once_with(
            config,
            Path("binary"),
            "a" * 64,
            policy_path=Path("policy.json"),
            approved_sha="b" * 64,
        )


if __name__ == "__main__":
    unittest.main()
