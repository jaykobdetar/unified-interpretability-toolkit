"""Strict callers bind canonical coordinator types and compatibility exports."""

from pathlib import Path
from typing import Any
from atlas_host import live_inference as canonical
from live_inference import Session, Snapshot, TickOwner, available


def snapshots(python: str, model: Path) -> tuple[Snapshot, canonical.Snapshot]:
    return (
        Session(python, model).snapshot(),
        canonical.Session(python, model).snapshot(),
    )


def start(session: Session, request: dict[str, Any]) -> canonical.Snapshot:
    return session.start(request)


def transport(session: TickOwner) -> tuple[int, bytes, str]:
    return canonical.proxy_request(8797, "GET", "/api/model", session)


def memory_reader() -> int:
    return available()
