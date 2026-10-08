"""Strict clients retain service callback, record and handler return contracts."""

from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
from typing import Any, IO

from analytics.service import AnalyticsJobs, Metadata, Snapshot, route


def create(
    python: str,
    model: Path,
    busy: Callable[[], bool],
    fetch: Callable[[], Mapping[str, Any]],
    available: Callable[[], int],
    reap: Callable[[subprocess.Popen[bytes]], bool],
    clock: Callable[[], float],
) -> AnalyticsJobs:
    return AnalyticsJobs(
        python,
        model,
        inference_busy=busy,
        fetch_model=fetch,
        available=available,
        reap=reap,
        clock=clock,
    )


def records(jobs: AnalyticsJobs) -> tuple[Metadata, Snapshot]:
    return jobs.metadata(), jobs.snapshot()


class StringHandler:
    def __init__(self, stream: IO[bytes]) -> None:
        self.rfile = stream
        self.command = "GET"

    def send(self, status: int, body: object) -> str:
        return str(status)


def reply(handler: StringHandler, jobs: AnalyticsJobs) -> str:
    return route(handler, "/api/analytics", 0, jobs)
