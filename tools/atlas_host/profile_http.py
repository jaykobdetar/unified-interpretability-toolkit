"""Loopback profile bridge installed only by the explicit owner profile launcher."""

from urllib.parse import urlsplit, parse_qs, urlencode
from typing import Any, Protocol, cast
from .startup_diagnostics import diagnose
from .common import canonical, require
from .config import LOCAL_LIMITS
from .profile_os import strict_json
from .runtime_adapter import (
    FixtureHost,
    NativeResponse,
    OwnedRenderer,
    dispatch,
    fixture_entry,
)
from .hosted_runtime import HostedApplication
from .static_operation import StaticOperation
from host_atlas import HostHandler, ROOT, BUNDLE


class HostedServer(Protocol):
    application: HostedApplication
    host: FixtureHost


class HostedHandler(HostHandler):
    def handle(self) -> None:
        app = cast(HostedServer, self.server).application
        grant, meter = app.profiles.admission()  # Before receive/framing/body work.
        self.profile_admission = grant
        try:
            super().handle()
        finally:
            try:
                operation = cast(
                    StaticOperation | None, getattr(self, "static_finalizer", None)
                )
                if operation is not None and not operation.finished:
                    operation.abort()
            finally:
                meter.freeze()
                app.profiles.finish_admission(grant)

    def dispatch_host(
        self, method: str, raw_path: str, data: dict[str, Any] | None = None
    ) -> NativeResponse:
        app = cast(HostedServer, self.server).application
        operation = None
        parsed = urlsplit(raw_path)
        parts = parsed.path.strip("/").split("/")
        if (
            getattr(app, "dense_policy", None) is not None
            or getattr(app.host, "dense_policy", None) is not None
        ):
            kind = None
            if (
                method == "POST"
                and parts == ["api", "view-contexts"]
                and not parsed.query
            ):
                # Classify from trusted registry state, never the picker hint.
                # A denied/stale hint cannot select an ungated reuse path.
                if type(data) is dict and "model_id" in data:
                    entry = app.host.registry.owner_receipt(data["model_id"])
                    if not fixture_entry(entry):
                        kind = "metadata"
            elif (
                method == "GET"
                and len(parts) == 4
                and parts[:2] == ["api", "models"]
                and app.host.view_kind == "static"
            ):
                kind = {
                    "inspect": "inspect",
                    "tile": "tile",
                    "model": "metadata",
                    "progress": "metadata",
                    "tensor-status": "metadata",
                    "view": "metadata",
                }.get(parts[3])
            if kind is not None:
                require(
                    getattr(self, "static_finalizer", None) is None,
                    "Static operation already owned",
                )
                operation = app.begin_static(self.profile_admission, kind)
                self.static_finalizer = operation
        return dispatch(app.host, method, raw_path, data, operation=operation)

    def send(self, status: int, body: object, mime: str = "application/json") -> None:
        if getattr(self, "static_write_failed", False):
            return
        operation = cast(
            StaticOperation | None, getattr(self, "static_finalizer", None)
        )
        if operation is None or operation.finished:
            return super().send(status, body, mime)
        if status >= 400:
            operation.abort()
            return super().send(status, body, mime)
        try:
            # Serialization and bounded socket writing share the original grant.
            if type(body) is not bytes:
                body = canonical(body)
            require(
                len(body) <= LOCAL_LIMITS["upstream_response_bytes"],
                "Static response exceeds bound",
            )
            operation.check()
            self.connection.settimeout(
                min(
                    LOCAL_LIMITS["write_deadline_ms"] / 1000,
                    operation.grant.remaining()["wall_ms"] / 1000,
                )
            )
            return operation.publish(
                lambda: super(HostedHandler, self).send(status, body, mime)
            )
        except BaseException:
            operation.abort()
            self.static_write_failed = True
            self.close_connection = True
            raise

    def handle_action(self) -> None:
        app = cast(HostedServer, self.server).application
        try:
            path, length = self.validated()
            for key in (
                "Host",
                "Origin",
                "X-Atlas-Local",
                "Content-Type",
                "Sec-Fetch-Site",
            ):
                require(
                    len(self.headers.get_all(key, [])) <= 1, "Duplicate guarded header"
                )
            if path.startswith("/api/profiles/"):
                require(
                    getattr(self.server, "profile_controls", False) is True,
                    "Profile HTTP controls disabled",
                )
                require(
                    self.command == "POST" and self.path == path,
                    "Private POST controls required",
                )
                require(
                    self.headers.get("Origin")
                    == "http://" + self.headers.get("Host", "")
                    and self.headers.get("X-Atlas-Local") == "1"
                    and self.headers.get("Content-Type", "").split(";")[0]
                    == "application/json",
                    "Same-origin local JSON control required",
                )
                require(0 < length <= 8192, "Private body exceeds bound")
                action = path.rsplit("/", 1)[-1]
                status, body, _ = app.api.handle(
                    action,
                    strict_json(self.rfile.read(length)),
                    admission=self.profile_admission if action == "start" else None,
                )
                return self.send(status, body)
            if self.command == "GET" and path == "/api/models":
                require(not length and self.path == path, "Catalog request unavailable")
                with app.lock:
                    catalog = app.host.catalog()
                catalog.update(
                    profiles_enabled=getattr(self.server, "profile_controls", False)
                    is True,
                    resume_available=False,
                )
                return self.send(200, catalog)
            if self.command == "GET" and path == "/viewer.js":
                require(not length and self.path == path, "Asset request unavailable")
                bundle = list(BUNDLE)
                if getattr(self.server, "profile_controls", False) is True:
                    bundle.insert(bundle.index("host-client.js"), "profile-client.js")
                return self.send(
                    200,
                    b"\n;\n".join(
                        (ROOT / "web" / name).read_bytes() for name in bundle
                    ),
                    "text/javascript",
                )
            if (
                self.command == "GET"
                and path.startswith("/api/models/")
                and path.endswith("/binding")
            ):
                parsed = urlsplit(self.path)
                parts = path.strip("/").split("/")
                query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=4)
                require(
                    len(parts) == 4
                    and set(query) <= {"context", "tensor", "slice"}
                    and "context" in query
                    and all(len(v) == 1 for v in query.values()),
                    "Binding query",
                )
                with app.lock:
                    app.host._validate_context(parts[2], query["context"][0])
                    require(
                        getattr(app.host, "view_kind", None) != "static",
                        "Profiles unavailable for static views",
                    )
                    native = "/api/binding?" + urlencode(
                        {k: v[0] for k, v in query.items() if k != "context"}
                    )
                    status, raw, mime = cast(OwnedRenderer, app.host.reader).read(
                        native
                    )
                    require(
                        status == 200 and mime == "application/json",
                        "Binding unavailable",
                    )
                    body = strict_json(raw)
                    app.contexts.remember(
                        parts[2], query["context"][0], body["source_binding"]
                    )
                    return self.send(200, body)
            # Shared host lock serializes model switches with authorization/page
            # starts. A busy profile refuses source replacement/release; lease
            # expiry and cancel still have independent service execution.
            with app.lock:
                if (
                    self.command == "POST"
                    and path.startswith("/api/view-contexts")
                    and not path.endswith("/heartbeat")
                ):
                    require(
                        not app.supervisor.busy()
                        and app.supervisor.profile_session is None,
                        "Cancel and confirm profile cleanup before replacing or releasing source",
                    )
                if (
                    self.command == "GET"
                    and path.startswith("/api/models/")
                    and path.endswith("/view")
                ):
                    status, raw, mime = self.dispatch_host("GET", self.path)
                    body = strict_json(raw)
                    context = body["host_context"]
                    if getattr(app.host, "view_kind", None) != "static":
                        app.contexts.remember(
                            context["model_id"],
                            context["context_id"],
                            body["source_binding"],
                        )
                    return self.send(status, body, mime)
                cast(HostedServer, self.server).host = app.host
                return super().handle_action()
        except (ValueError, KeyError, TypeError, OSError) as error:
            diagnose(
                getattr(self.server, "application", None),
                "profile_http.request_refused",
                error,
            )
            return self.send(
                409,
                {
                    "version": 1,
                    "code": "unavailable",
                    "error": "Hosted request unavailable or cleanup pending",
                },
            )

    do_GET = handle_action
    do_POST = handle_action
