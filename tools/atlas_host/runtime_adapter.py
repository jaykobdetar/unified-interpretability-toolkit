"""One owned reader with tab leases; registered static admission defaults closed."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from .common import canonical, digest, fields, require
from .registry import fingerprint
from .static_models import StaticPolicy

FIXTURE_SHA = "c0075bfc55f9e51ccac3c5511ea55a5ca19744b002921e8d2e4ae3f60d321be3"
FIXTURE_FILES = [{"name": "tiny.safetensors", "bytes": 244, "sha256": FIXTURE_SHA}]
FIXTURE_TENSORS = {"matrix": [3, 5], "vector": [7], "zeros": [4]}
READ_ROUTES = {
    "model": ("/api/model", set()),
    "progress": ("/api/progress", {"tensor"}),
    "tensor-status": ("/api/tensor-status", {"tensor"}),
    "view": ("/api/view", {"tensor", "slice", "left", "right"}),
    "inspect": ("/api/inspect", {"tensor", "slice", "row", "col", "left", "right"}),
    "tile": ("/tile", {"tensor", "slice", "rule", "level", "x", "y"}),
}


class HostError(ValueError):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def refuse(status, code, message):
    raise HostError(status, code, message)


def fixture_entry(entry):
    return (
        entry["manifest"]["provenance"] == "synthetic_fixture"
        and entry["manifest"]["files"] == FIXTURE_FILES
    )


def check_fixture(entry, *, hash_bytes=False):
    require(
        entry["enabled"] and fixture_entry(entry),
        "Only enabled exact synthetic fixtures may activate",
    )
    root = Path(entry["root"])
    # Match all files the source loader could interpret, not only the listed shard.
    interpreted = {
        path.name
        for path in root.iterdir()
        if path.name.endswith(".safetensors")
        or path.name == "model.safetensors.index.json"
    }
    require(interpreted == {"tiny.safetensors"}, "Fixture source inventory changed")
    path = root / "tiny.safetensors"
    before = fingerprint(path.lstat())
    require(
        before == entry["fingerprints"]["tiny.safetensors"],
        "Fixture fingerprint changed",
    )
    if hash_bytes:
        # Exactly 244 known fixture bytes, not an arbitrary model hash request.
        import os

        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            require(
                fingerprint(os.fstat(source.fileno())) == before,
                "Fixture source changed",
            )
            raw = source.read(245)
            require(
                len(raw) == 244 and hashlib.sha256(raw).hexdigest() == FIXTURE_SHA,
                "Fixture hash differs from the allowed synthetic source",
            )
        require(
            fingerprint(path.lstat()) == before, "Fixture changed during activation"
        )


class FixtureHost:
    """factory(entry, cache_path) returns an owned renderer handle.

    Handle methods: ready(), alive(), read(path)->(status,bytes,mime), stop()->bool.
    Optional initialize() runs only after the handle is owned by this host slot.
    Tests inject fake handles; only the separate launcher provides real processes.
    Every method is called by the existing serial coordinator, not HTTP threads.
    """

    def __init__(
        self,
        registry,
        cache_root,
        factory,
        *,
        clock=time.monotonic,
        static_policy=None,
        static_admission=None,
        dense_policy=None,
    ):
        require(
            static_policy is None
            or type(static_policy) is StaticPolicy
            and static_policy.registry is registry,
            "Same-registry private static policy required",
        )
        require(
            static_admission is None
            or static_policy is not None
            and callable(static_admission),
            "Private static admission requires its policy",
        )
        self.registry, self.cache_root, self.factory = (
            registry,
            Path(cache_root).resolve(),
            factory,
        )
        from .dense_static_admission import BoundDenseStaticAdmission

        require(
            dense_policy is None
            or type(dense_policy) is BoundDenseStaticAdmission
            and dense_policy.registry is registry,
            "Same-registry sealed dense policy required",
        )
        self.dense_policy = dense_policy
        self.static_policy, self.static_admission = static_policy, static_admission
        self.clock = clock
        self.entry = self.reader = None
        self.context = None
        self.leases = {}
        self.stopping = False
        self.started = 0.0
        self.source_identity = self.model_identity = None
        self.view_kind = None
        self.static_prepared = self.static_bound = None
        self.observed_alive = self.observed_ready = False
        self.observed_context = None

    @property
    def static_views_enabled(self):
        return self.static_policy is not None and self.static_admission is not None

    def _revoke_readiness(self):
        self.source_identity = self.model_identity = None
        self.static_prepared = self.static_bound = None
        self.observed_alive = self.observed_ready = False
        self.observed_context = None

    def _published_ready(self):
        # Read cached coordinator observations only. alive()/ready() can reap.
        return bool(
            self.reader is not None
            and not self.stopping
            and self.context is not None
            and self.observed_context == self.context
            and self.observed_alive
            and self.observed_ready
            and self.source_identity is not None
            and self.model_identity is not None
            and any(end > self.clock() for end in self.leases.values())
            and (self.view_kind != "static" or self.static_bound is not None)
        )

    def _observe_reader(self):
        alive = self.reader.alive()
        ready = self.reader.ready() if alive else False
        require(
            type(alive) is bool and type(ready) is bool, "Invalid owned reader health"
        )
        self.observed_alive, self.observed_ready = alive, ready
        self.observed_context = self.context

    def _check_source(self):
        fresh = self.registry.owner_receipt(self.entry["model_id"])
        require(fresh == self.entry, "Installed receipt changed")
        if self.view_kind == "static":
            require(self.static_prepared is not None, "Static preparation revoked")
            self.static_prepared.check()
        else:
            check_fixture(self.entry)

    def _clear(self):
        self.entry = self.reader = self.context = None
        self.leases.clear()
        self.stopping = False
        self.view_kind = None
        self._revoke_readiness()

    def _stop(self):
        self.stopping = True
        self.leases.clear()
        self._revoke_readiness()  # Revoke before fallible/uncertain stop.
        if self.reader is None or self.reader.stop():
            self._clear()
            return True
        return False

    def tick(self):
        if self.reader is None:
            return
        if self.stopping:
            self._stop()
            return
        now = self.clock()
        self.leases = {cap: end for cap, end in self.leases.items() if end > now}
        if not self.leases:
            self._stop()
            return
        try:
            self._check_source()
            self._observe_reader()
        except Exception:
            self._stop()
            return
        if (
            not self.observed_alive
            or (self.source_identity is not None and not self.observed_ready)
            or (self.source_identity is None and now - self.started >= 5)
        ):
            self._stop()

    def close(self):
        return self._stop()

    def catalog(self):
        # No tick/reap/start or payload reads from this read-only endpoint.
        ready = self._published_ready()
        catalog = (
            self.static_policy.catalog(
                active_binding=self.static_bound, reader_ready=ready
            )
            if self.static_policy is not None
            else self.registry.catalog()
        )
        for item in catalog["models"]:
            item.setdefault("static_view_candidate", False)
            item.setdefault("static_view_ready", False)
            item["static_activation_allowed"] = bool(
                self.dense_policy is not None
                and self.dense_policy.allows(item["model_id"])
            )
            item["fixture_eligible"] = False
            if item["hash_provenance"] == "synthetic_fixture":
                entry = self.registry.owner_receipt(item["model_id"])
                current = self.registry._unchanged(entry)
                item["state"] = (
                    "verified_pending_renderer" if current else "source_changed"
                )
                item["fixture_eligible"] = fixture_entry(entry) and current
                item["view_ready"] = bool(
                    item["fixture_eligible"] and entry == self.entry and ready
                )
            item["reader_state"] = (
                ("stopping" if self.stopping else "active")
                if (self.entry and self.entry["model_id"] == item["model_id"])
                else "closed"
            )
        return {
            "api_version": 1,
            **catalog,
            "reader_capacity": 1,
            "lease_capacity": 4,
            "static_views_enabled": self.static_views_enabled,
            "inference_enabled": False,
            "downloads_enabled": False,
            "profiles_enabled": False,
        }

    def _owns(self, context, capability):
        return (
            context == self.context
            and type(capability) is str
            and any(
                secrets.compare_digest(cap.encode(), capability.encode())
                for cap in self.leases
            )
        )

    def acquire(self, data, *, operation=None):
        fields(data, ("model_id",), ("context_id", "capability"))
        require(
            ("context_id" in data) == ("capability" in data),
            "Complete current lease required",
        )
        if operation is not None:
            operation.check()
        self.tick()
        if operation is not None:
            operation.check()
        if self.stopping:
            refuse(503, "cleanup_pending", "Reader cleanup is pending")
        current = data.get("capability")
        if current is not None and not self._owns(data["context_id"], current):
            refuse(409, "stale_context", "Current reader lease is unavailable")
        kind = None
        try:
            entry = self.registry.owner_receipt(data["model_id"])
            kind = "fixture" if fixture_entry(entry) else "static"
            if kind == "fixture":
                check_fixture(entry, hash_bytes=True)
            else:
                if not self.static_views_enabled:
                    refuse(
                        409,
                        "static_disabled",
                        "Registered static activation is unavailable",
                    )
        except HostError:
            raise
        except (ValueError, OSError):
            if kind == "static":
                refuse(
                    409,
                    "static_unavailable",
                    "Registered static activation is unavailable",
                )
            refuse(
                409,
                "fixture_unavailable",
                "Enabled verified synthetic fixture required",
            )
        if self.reader is not None and self.entry["model_id"] != entry["model_id"]:
            if current is None or len(self.leases) != 1:
                refuse(
                    409,
                    "reader_busy",
                    "Another tab owns the reader; close its view or wait for its lease",
                )
        prepared = None
        if kind == "static":
            try:
                if self.dense_policy is not None:
                    from .hosted_runtime import StaticOperation

                    require(
                        type(operation) is StaticOperation
                        and operation.app.host is self
                        and operation.app.dense_policy is self.dense_policy
                        and operation.token.kind == "metadata"
                        and operation.token.current(),
                        "Original private dense acquisition required",
                    )
                if operation is not None:
                    operation.check()
                prepared = (
                    self.static_prepared
                    if self.reader is not None and self.entry == entry
                    else self.static_policy.prepare(data["model_id"])
                )
                require(prepared is not None, "Current static preparation required")
                if operation is not None:
                    operation.prepared = prepared
                    operation.check()
                require(
                    prepared.entry == entry, "Static receipt changed before admission"
                )
                require(
                    self.static_admission(prepared) is True,
                    "Static owner/resource admission refused",
                )
                prepared.check()
            except (ValueError, OSError):
                refuse(
                    409,
                    "static_unavailable",
                    "Registered static activation is unavailable",
                )
        if self.reader is not None and self.entry["model_id"] != entry["model_id"]:
            if operation is not None:
                operation.mutated = True
                operation.check()
            if not self._stop():
                refuse(503, "cleanup_pending", "Previous reader cleanup is pending")
            if operation is not None:
                operation.check()
            current = None
        if self.reader is None:
            cache = self.cache_root / entry["model_id"]
            require(
                not cache.is_relative_to(Path(entry["root"])),
                "Cache must be outside source",
            )
            try:
                reader = (
                    self.factory(
                        deepcopy(entry), cache, prepared=prepared, operation=operation
                    )
                    if kind == "static" and operation is not None
                    else self.factory(deepcopy(entry), cache)
                )
            except (ValueError, OSError):
                refuse(503, "activation_failed", "Fixture renderer could not start")
            # Own the child before any fallible pipe/readiness initialization.
            self.reader = reader
            self.entry = entry
            self.view_kind, self.static_prepared = kind, prepared
            try:
                if operation is not None:
                    operation.check()
                initialize = getattr(reader, "initialize", None)
                if initialize is not None:
                    initialize()
                if operation is not None:
                    operation.check()
            except BaseException as error:
                try:
                    self._stop()
                except Exception:
                    # Uncertain cleanup keeps the owned stopping handle/slot.
                    # Preserve the setup diagnostic instead of replacing it.
                    pass
                if not isinstance(error, Exception):
                    raise
                raise HostError(
                    503, "activation_failed", "Fixture renderer could not start"
                ) from error
            self.context, self.started = secrets.token_hex(16), self.clock()
        if current is None:
            if len(self.leases) >= 4:
                refuse(409, "reader_busy", "Reader tab lease capacity reached")
            current = secrets.token_hex(32)
        self.leases[current] = self.clock() + 15
        if kind == "static" and operation is not None:
            operation.context = self.context
            if self.static_bound is None:
                # Pending lease stays private until complete native binding.
                self.read(
                    entry["model_id"],
                    "model",
                    {"context": [self.context]},
                    operation=operation,
                )
            operation.check()
        return {
            "api_version": 1,
            "model_id": entry["model_id"],
            "context_id": self.context,
            "capability": current,
            "lease_seconds": 15,
            "view_kind": kind,
            **({"profiles_enabled": False} if kind == "static" else {}),
            "state": "active" if self.source_identity else "starting",
        }

    def heartbeat(self, context, data):
        fields(data, ("capability",))
        self.tick()
        if not self._owns(context, data["capability"]):
            refuse(409, "stale_context", "Reader lease expired or is not owned")
        self.leases[data["capability"]] = self.clock() + 15
        return {"api_version": 1, "context_id": context, "lease_seconds": 15}

    def release(self, context, data):
        fields(data, ("capability",))
        self.tick()
        if not self._owns(context, data["capability"]):
            refuse(409, "stale_context", "Reader lease expired or is not owned")
        del self.leases[data["capability"]]
        if not self.leases:
            self._stop()
        return {"api_version": 1, "released": True, "cleanup_pending": self.stopping}

    def _validate_context(self, identifier, context, *, allow_unbound_model=False):
        # Preserve the fixture's source_changed receipt before tick can dispose
        # the old context. This is also the pre-channel static source check.
        if (
            not self.stopping
            and self.entry is not None
            and self.context == context
            and self.entry["model_id"] == identifier
        ):
            try:
                self._check_source()
            except (ValueError, OSError):
                self._stop()
                refuse(
                    409,
                    "source_changed",
                    "Installed source changed; reopen after owner verification",
                )
        self.tick()
        if (
            self.stopping
            or self.entry is None
            or self.context != context
            or self.entry["model_id"] != identifier
        ):
            refuse(409, "stale_context", "Selected model context is unavailable")
        try:
            self._check_source()
        except (OSError, ValueError):
            self._stop()
            refuse(
                409,
                "source_changed",
                "Installed fixture changed; reopen after owner verification",
            )
        if not self.observed_ready:
            refuse(503, "backend_unavailable", "Fixture renderer is starting")
        if (
            self.view_kind == "static"
            and not allow_unbound_model
            and self.static_bound is None
        ):
            refuse(
                409,
                "context_not_ready",
                "Read selected model metadata before requesting values",
            )

    def _bind_model(self, model):
        if self.view_kind == "static":
            require(
                self.observed_alive
                and self.observed_ready
                and self.observed_context == self.context
                and not self.stopping
                and any(end > self.clock() for end in self.leases.values()),
                "Current owned ready reader required",
            )
            if self.dense_policy is not None:
                from .dense_static_admission import check_native_model

                check_native_model(self.static_prepared, model)
            bound = self.static_policy.bind(
                self.static_prepared, model, reader_ready=True
            )
            self.static_bound = bound
            self.source_identity = bound.prepared.source_identity
            self.model_identity = bound.prepared.model_identity
            return
        require(
            type(model) is dict and model.get("api_version") == 1,
            "Invalid renderer metadata",
        )
        digest(model.get("source_identity"))
        digest(model.get("model_identity"))
        require(
            model.get("source_directory") == self.entry["root"]
            and model.get("revision") == self.entry["manifest"]["revision"]
            and model.get("source_bytes") == 244
            and model.get("parameter_count") == 26,
            "Renderer differs from selected receipt",
        )
        expected_model = hashlib.sha256(
            canonical(
                ["weight-atlas-model-v1", model["source_identity"], model["revision"]]
            )
        ).hexdigest()
        require(
            model["model_identity"] == expected_model,
            "Renderer revision binding differs",
        )
        tensors = model.get("catalog")
        require(
            type(tensors) is list and len(tensors) == 3, "Unexpected fixture catalog"
        )
        require(
            {t.get("name") for t in tensors} == set(FIXTURE_TENSORS)
            and {t.get("id") for t in tensors} == {0, 1, 2},
            "Unexpected fixture tensor identity",
        )
        for tensor in tensors:
            require(
                tensor.get("shape") == FIXTURE_TENSORS[tensor["name"]]
                and tensor.get("dtype") == "BF16"
                and tensor.get("shard") == "tiny.safetensors",
                "Unexpected fixture native storage",
            )
        if self.source_identity is not None:
            require(
                model["source_identity"] == self.source_identity
                and model["model_identity"] == self.model_identity,
                "Renderer source changed",
            )
        self.source_identity, self.model_identity = (
            model["source_identity"],
            model["model_identity"],
        )

    def read(self, identifier, route, query, *, operation=None):
        require(route in READ_ROUTES, "Unknown model read route")
        native, keys = READ_ROUTES[route]
        require(
            set(query) <= keys | {"context"}
            and "context" in query
            and all(type(v) is list and len(v) == 1 for v in query.values()),
            "Invalid query fields",
        )
        context = query["context"][0]
        if operation is not None:
            operation.check()
        self._validate_context(
            identifier, context, allow_unbound_model=route == "model"
        )
        if operation is not None:
            operation.prepared = self.static_prepared
            operation.context = context
            operation.check()
        if route != "model" and self.source_identity is None:
            refuse(
                409,
                "context_not_ready",
                "Read selected model metadata before requesting values",
            )
        if route == "tensor-status":
            require("tensor" in query, "Selected tensor required")
        native_query = {k: values[0] for k, values in query.items() if k != "context"}
        path = native + ("?" + urlencode(native_query) if native_query else "")
        owned_reader = self.reader
        try:
            status, body, mime = (
                owned_reader.read(path, operation=operation)
                if operation is not None
                else owned_reader.read(path)
            )
            if operation is not None:
                operation.check()
            self._validate_context(
                identifier, context, allow_unbound_model=route == "model"
            )
            require(
                type(body) is bytes
                and len(body)
                <= (16384 if route in ("progress", "tensor-status") else 2 * 1024**2),
                "Renderer response exceeds bound",
            )
            if status != 200:
                refuse(
                    503 if status >= 500 else 400,
                    "renderer_unavailable",
                    "Requested model data is unavailable",
                )
            if route == "tile":
                require(mime == "image/png", "Unexpected tile response")
                return status, body, mime
            require(mime == "application/json", "Unexpected renderer metadata response")
            from .profile_os import strict_json

            model = strict_json(body)
            if route == "model":
                self._bind_model(model)
                if self.view_kind == "static":
                    model = self.static_policy.project_model(
                        self.static_bound, model, reader_ready=self._published_ready()
                    )
                    model["profiles_enabled"] = False
                else:
                    for key in (
                        "source_directory",
                        "inference_source_model",
                        "head_layout",
                        "head_layout_binding",
                        "fresh_source_hashes",
                    ):
                        model.pop(key, None)
                    model["inference_editable"] = False
                    model["content_digest"] = self.entry["content_digest"]
                    model["identity_validation"] = (
                        "Owner-registered synthetic fixture; current file identity checked. "
                        "Saved calibration is separate from source hash verification."
                    )
                model["view_kind"] = self.view_kind
            elif route in ("progress", "tensor-status"):
                require(
                    model.get("source_identity") == self.source_identity
                    and model.get("model_identity") == self.model_identity,
                    "Status source mismatch",
                )
            if self.view_kind == "static" and route in ("view", "inspect"):
                require(
                    type(model.get("source_binding")) is dict
                    and model["source_binding"].get("source_identity")
                    == self.source_identity
                    and model["source_binding"].get("model_identity")
                    == self.model_identity,
                    "Static value source mismatch",
                )
            model["host_context"] = {
                "model_id": identifier,
                "context_id": context,
                "source_identity": self.source_identity,
                "model_identity": self.model_identity,
            }
        except BaseException as error:
            if self.reader is owned_reader and self.context == context:
                try:
                    self._stop()
                except Exception:
                    pass  # Uncertain cleanup retains ownership and revoked readiness.
            if not isinstance(error, Exception):
                raise
            if isinstance(error, HostError):
                raise
            refuse(
                409,
                "source_changed",
                "Renderer receipt correspondence could not be verified",
            )
        raw = canonical(model)
        if operation is not None:
            operation.check()
        return 200, raw, "application/json"


def dispatch(host, method, raw_path, data=None, *, operation=None):
    """Pure handler router; HTTP guards run in the coordinator before this call."""
    parsed = urlsplit(raw_path)
    require(
        not parsed.scheme and not parsed.netloc and not parsed.fragment,
        "Local route required",
    )
    parts = parsed.path.strip("/").split("/")
    query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=12)
    if method == "GET" and parts == ["api", "models"] and not query:
        return 200, canonical(host.catalog()), "application/json"
    if method == "POST" and parts == ["api", "view-contexts"] and not query:
        return (
            202,
            canonical(
                host.acquire(data, operation=operation)
                if operation is not None
                else host.acquire(data)
            ),
            "application/json",
        )
    if (
        method == "POST"
        and len(parts) == 4
        and parts[:2] == ["api", "view-contexts"]
        and not query
    ):
        action = {"heartbeat": host.heartbeat, "release": host.release}.get(parts[3])
        if action:
            return 200, canonical(action(parts[2], data)), "application/json"
    if method == "GET" and len(parts) == 4 and parts[:2] == ["api", "models"]:
        return (
            host.read(parts[2], parts[3], query, operation=operation)
            if operation is not None
            else host.read(parts[2], parts[3], query)
        )
    refuse(404, "not_found", "Route is unavailable in fixture host mode")
