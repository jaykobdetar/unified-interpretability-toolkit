"""UNREGISTERED private profile router. Existing HTTP hosts remain unchanged.

The future hosted handler must create an admission grant before parsing a start
body. Its executor owns construction/start/ticks independently of HTTP requests.
This module neither starts an executor nor enables profiles in any catalog.
"""

from copy import deepcopy

from .cache import binding
from .common import canonical, digest, fields, integer, require

ROUTES = {"start", "status", "page", "cancel", "heartbeat", "reconcile"}
BASE = ("version", "model_id", "context_id", "tab_capability")
OWNER = BASE + ("job_id", "job_capability")


def production_capabilities():
    # Only a separately reviewed activation change may alter this value.
    return {"profiles_enabled": False, "resume_available": False}


class PrivateProfileAPI:
    """service is a hosted owner-executor bridge, never a ProfileJob itself.

    service.start(validated request, admission) reserves immediately or refuses;
    service.status/page are immutable/pinned reads; cancel is signal-only.
    No generic retry or poll-driven tick. Service must validate live tab ownership
    before exposing even terminal records and bind each job to model/context.
    """

    def __init__(self, service, authorize_tab):
        self.service = service
        self.authorize_tab = authorize_tab

    def handle(self, action, data, *, admission=None):
        require(action in ROUTES, "Profile route unavailable")
        require(len(canonical(data)) <= 8192, "Profile request exceeds body limit")
        extra = ("binding", "seed", "values", "restart") if action == "start" else ()
        if action == "page":
            extra = ("revision", "axis", "start", "count")
        fields(
            data, BASE + extra if action in ("start", "reconcile") else OWNER + extra
        )
        integer(data["version"], 1, 1)
        require(
            type(data["model_id"]) is str and data["model_id"].startswith("m_"),
            "Invalid model",
        )
        digest(data["model_id"][2:])
        require(
            type(data["context_id"]) is str and bool(data["context_id"]),
            "Context required",
        )
        digest(data["tab_capability"])
        self.authorize_tab(data["model_id"], data["context_id"], data["tab_capability"])
        if action == "start":
            require(admission is not None, "HTTP-anchored admission required")
            admission.remaining()
            selected = binding(data["binding"])
            integer(data["seed"], 0, 2**32 - 1)
            integer(data["values"], 1, selected["rows"] * selected["cols"])
            require(type(data["restart"]) is bool, "Explicit Restart intent required")
            # Owner service compares selected to renderer/registry correspondence.
            result = self.service.start(deepcopy(data), admission)
            status = 202
        elif action == "reconcile":
            require(admission is None, "Reconciliation cannot grant work")
            result = self.service.reconcile(deepcopy(data))
            status = 200
        else:
            require(admission is None, "Only explicit start receives a grant")
            digest(data["job_capability"])
            require(
                type(data["job_id"]) is str
                and len(data["job_id"]) == 32
                and all(c in "0123456789abcdef" for c in data["job_id"]),
                "Invalid job ID",
            )
            if action == "page":
                digest(data["revision"])
                require(data["axis"] in ("rows", "columns"), "Invalid profile axis")
                integer(data["start"])
                integer(data["count"], 1, 1024)
            result = getattr(self.service, action)(deepcopy(data))
            status = 200
        maximum = 2 * 1024**2 if action == "page" else 16384
        raw = canonical(result)
        require(len(raw) <= maximum, "Profile response exceeds bound")
        # Capability delivery occurs only in the initial private start response.
        if action != "start":
            require(
                data["tab_capability"].encode() not in raw
                and (
                    action == "reconcile" or data["job_capability"].encode() not in raw
                ),
                "Private capability in result",
            )
        return status, result, {"Cache-Control": "no-store"}
