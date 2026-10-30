"""Per-check results from the grader, plus a bounded reward file."""

import json
from pathlib import Path

import pytest

CHECKS = Path("/logs/verifier/checks.json")
NAMES = (
    "compile_submission",
    "probe_unknown_type",
    "probe_truncated_frame",
    "bulk_book",
    "bulk_rss",
    "serving_book",
    "serving_rss",
    "backfill_book",
    "backfill_rss",
    "spill_book",
    "spill_rss",
)


def test_reward_file_is_bounded():
    raw = Path("/logs/verifier/reward.txt").read_text().strip()
    value = float(raw)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("name", NAMES)
def test_grader_check(name: str):
    payload = json.loads(CHECKS.read_text())
    status = payload[name]
    if status == "skip":
        pytest.skip(name)
    assert status == "pass"
