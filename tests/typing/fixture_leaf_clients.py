"""Strict fixture-receipt callers; declarations perform no file or launch work."""

from typing import assert_type

from atlas_host.fixture_source import check_fixture, fixture_entry
from atlas_host.registry import RegistryEntry


def fixture_receipt(entry: RegistryEntry) -> bool:
    return assert_type(fixture_entry(entry), bool)


def recheck_fixture(entry: RegistryEntry, *, hash_bytes: bool = False) -> None:
    assert_type(check_fixture(entry, hash_bytes=hash_bytes), None)
