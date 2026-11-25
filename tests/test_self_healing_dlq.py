"""Triad unit tests for video container self-healing and Dead Letter Queue fault isolation."""

import json
import os
import tempfile
import pytest

from shorts_generator.engine.self_healing import CorruptedVideoError, SelfHealingVideoSanitizer


def test_inspect_container_rejects_missing_and_empty_files() -> None:
    """Negative: Verify missing or zero-byte files are immediately flagged as corrupt."""
    healthy, err = SelfHealingVideoSanitizer.inspect_container("non_existent_file_path.mp4")
    assert healthy is False
    assert "does not exist" in err

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    try:
        healthy_zero, err_zero = SelfHealingVideoSanitizer.inspect_container(temp_path)
        assert healthy_zero is False
        assert "zero byte length" in err_zero
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_route_to_dlq_quarantines_asset_and_emits_json_report() -> None:
    """Positive: Verify unrecoverable corrupted files are quarantined with structured diagnostic report."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_corrupted_file = os.path.join(temp_dir, "broken_macroblock.mp4")
        with open(fake_corrupted_file, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypmp42corrupt_moov_atom_bytes")

        dlq_dir = os.path.join(temp_dir, "custom_dlq")
        quarantined_path = SelfHealingVideoSanitizer.route_to_dlq(
            media_path=fake_corrupted_file,
            reason="Invalid MOOV atom header",
            dlq_dir=dlq_dir
        )

        assert os.path.exists(quarantined_path)
        assert os.path.dirname(quarantined_path) == os.path.abspath(dlq_dir)

        report_file = os.path.join(dlq_dir, "broken_macroblock_failure_report.json")
        assert os.path.exists(report_file)

        with open(report_file, "r", encoding="utf-8") as rf:
            data = json.load(rf)
            assert data["reason"] == "Invalid MOOV atom header"
            assert data["source_size_bytes"] == os.path.getsize(fake_corrupted_file)
            assert "timestamp" in data


def test_ensure_playable_or_heal_raises_corrupted_video_error_with_dlq_path() -> None:
    """Negative: Verify that an unrepairable garbage file raises CorruptedVideoError referencing the DLQ."""
    with tempfile.TemporaryDirectory() as temp_dir:
        garbage_file = os.path.join(temp_dir, "totally_garbage.mp4")
        with open(garbage_file, "wb") as f:
            f.write(b"NOT_A_VALID_CONTAINER_AT_ALL_9999999")

        dlq_dir = os.path.join(temp_dir, "test_dlq")
        with pytest.raises(CorruptedVideoError) as exc_info:
            SelfHealingVideoSanitizer.ensure_playable_or_heal(garbage_file, dlq_dir=dlq_dir)

        err = exc_info.value
        assert err.dlq_path is not None
        assert os.path.exists(err.dlq_path)
        assert "Quarantined to" in str(err)
