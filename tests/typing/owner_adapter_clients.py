"""Actual command and receipt clients for the owner adapters."""

import subprocess
from typing import Any, assert_type
from atlas_host import cli
from atlas_host.inference import (
    ConfigurationEvidence,
    HeadLayoutMetadata,
    configuration_evidence,
    head_layout_metadata,
)
from atlas_host.prepare_fixture import FixtureReceipt, FixtureRunner, prepare


def metadata_client(
    manifest: dict[str, Any], descriptor: dict[str, Any], selected: dict[str, Any]
) -> tuple[ConfigurationEvidence, HeadLayoutMetadata]:
    evidence = configuration_evidence(manifest, descriptor, selected)
    metadata = head_layout_metadata(manifest, descriptor, selected)
    assert_type(evidence, ConfigurationEvidence)
    assert_type(metadata, HeadLayoutMetadata)
    assert_type(metadata["head_layout_binding"]["source_identity"], str)
    return evidence, metadata


def fixture_client(config: dict[str, Any], identifier: str) -> FixtureReceipt:
    runner: FixtureRunner = subprocess.run
    receipt = prepare(config, identifier, run=runner)
    assert_type(receipt, FixtureReceipt)
    assert_type(receipt["fixture_calibrated"], bool)
    return receipt


def owner_client(arguments: list[str]) -> int:
    result = cli.main(arguments)
    assert_type(result, int)
    return result
