"""Strict callers of real snapshot streaming, publication and page boundaries."""

from collections.abc import Callable, Generator
from typing import Any, assert_type

from atlas_host.profile_snapshot import (
    PublicationOptions,
    SnapshotInfo,
    SnapshotPage,
    SnapshotPageRow,
    SnapshotStore,
    layout,
    validation_steps,
)


def streaming(
    fd: int,
    selected: dict[str, Any],
    seed: int,
    revision: str,
    deadline: float,
    clock: Callable[[], float],
) -> Generator[None, None, SnapshotInfo]:
    assert_type(layout(selected), tuple[int, int, bytes])
    return assert_type(
        validation_steps(fd, selected, seed, revision, deadline=deadline, clock=clock),
        Generator[None, None, SnapshotInfo],
    )


def publication(
    store: SnapshotStore, revision: str, options: PublicationOptions
) -> SnapshotInfo:
    assert_type(store.owned_storage_bytes, int)
    assert_type(
        store.publication_steps(revision, **options),
        Generator[None, None, SnapshotInfo],
    )
    return assert_type(store.publish(revision, **options), SnapshotInfo)


def page(
    store: SnapshotStore, revision: str, source_check: Callable[[], object]
) -> SnapshotPage:
    with store.page_handle(revision, expected_publication=1) as (fd, info):
        assert_type(fd, int)
        assert_type(info, SnapshotInfo)
    result = assert_type(
        store.page(revision, "rows", 0, 1, source_check=source_check), SnapshotPage
    )
    for row in result["original"]:
        assert_type(row, SnapshotPageRow)
        assert_type(row["mean_abs"], float | None)
    return result
