"""Exact current renderer rule IDs for the optional combined acceptance."""

EXPECTED_RULE_IDS = (
    "global_linear",
    "global_asinh",
    "tensor_linear",
    "tensor_asinh",
    "tensor_magnitude",
    "tensor_magnitude_asinh",
    "tensor_robust99",
    "tensor_signed_percentile",
)


def assert_render_rules(metadata):
    actual = tuple(rule["id"] for rule in metadata["rules"])
    assert actual == EXPECTED_RULE_IDS, ("Unexpected renderer rule IDs", actual)
