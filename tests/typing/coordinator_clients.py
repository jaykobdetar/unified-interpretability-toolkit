"""Strict callers use the actual coordinator and its structural boundaries."""

from pathlib import Path
import socket
import subprocess
from typing import Any

from live_inference import (
    DeadlineReader,
    DeadlineSocket,
    LayoutReceipt,
    OperationDeadline,
    Session,
    Snapshot,
    process_alive,
    proxy_request,
    signal_and_reap,
    verified_layout_receipt,
)


def snapshot(python: str, model: Path) -> Snapshot:
    return Session(python, model).snapshot()


def request(session: Session, data: dict[str, Any]) -> Snapshot:
    return session.start(data)


def transport(
    connection: socket.socket, session: Session
) -> tuple[DeadlineReader, DeadlineSocket]:
    deadline = OperationDeadline(session)
    return DeadlineReader(connection, deadline), DeadlineSocket(connection, deadline)


def reply(session: Session) -> tuple[int, bytes, str]:
    return proxy_request(8797, "GET", "/api/model", session)


def child(process: subprocess.Popen[bytes]) -> tuple[bool, bool]:
    return process_alive(process), signal_and_reap(process)


def receipt(model: Path) -> LayoutReceipt | None:
    return verified_layout_receipt(model, required=True)
