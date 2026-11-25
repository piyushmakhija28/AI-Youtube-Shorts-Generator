"""Triad unit tests for automated broadcast color matrix detection and correction."""

from typing import Dict
import pytest

from shorts_generator.video.color_matrix_fixer import ColorMatrixFixer


def test_color_matrix_detects_bt601_variants() -> None:
    """Positive: Verify that common BT.601 color space variants are correctly identified."""
    variants = ["smpte170m", "bt470bg", "bt470_6m", "bt601", "bt601-6-525", "bt601-6-625"]
    for v in variants:
        meta = {"color_space": v, "color_transfer": "", "color_primaries": ""}
        assert ColorMatrixFixer.is_bt601(meta) is True

    transfer_meta = {"color_space": "unknown", "color_transfer": "smpte170m", "color_primaries": ""}
    assert ColorMatrixFixer.is_bt601(transfer_meta) is True

    primaries_meta = {"color_space": "unknown", "color_transfer": "", "color_primaries": "bt470bg"}
    assert ColorMatrixFixer.is_bt601(primaries_meta) is True


def test_color_matrix_generates_scale_filter_for_limited_and_full_range() -> None:
    """Positive: Verify correct FFmpeg scale filter expressions for limited and full range BT.601 inputs."""
    limited_meta = {
        "color_space": "smpte170m",
        "color_transfer": "smpte170m",
        "color_primaries": "smpte170m",
        "color_range": "tv",
    }
    filter_expr = ColorMatrixFixer.build_color_matrix_filter(limited_meta)
    assert filter_expr == "scale=in_color_matrix=bt601:out_color_matrix=bt709:in_range=limited:out_range=limited"

    full_meta = {
        "color_space": "bt470bg",
        "color_transfer": "bt470bg",
        "color_primaries": "bt470bg",
        "color_range": "pc",
    }
    full_filter = ColorMatrixFixer.build_color_matrix_filter(full_meta)
    assert full_filter == "scale=in_color_matrix=bt601:out_color_matrix=bt709:in_range=full:out_range=limited"


def test_color_matrix_bypasses_bt709_and_unspecified() -> None:
    """Negative: Verify that BT.709 or modern broadcast color spaces bypass filter generation."""
    bt709_meta = {
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "color_range": "tv",
    }
    assert ColorMatrixFixer.is_bt601(bt709_meta) is False
    assert ColorMatrixFixer.build_color_matrix_filter(bt709_meta) is None

    unknown_meta = {
        "color_space": "unspecified",
        "color_transfer": "unknown",
        "color_primaries": "reserved",
    }
    assert ColorMatrixFixer.is_bt601(unknown_meta) is False
    assert ColorMatrixFixer.build_color_matrix_filter(unknown_meta) is None


def test_color_matrix_empty_and_malformed_metadata_edge_cases() -> None:
    """Edge Case: Verify graceful handling of empty or malformed metadata structures."""
    empty_meta: Dict[str, str] = {}
    assert ColorMatrixFixer.is_bt601(empty_meta) is False
    assert ColorMatrixFixer.build_color_matrix_filter(empty_meta) is None

    tags = ColorMatrixFixer.get_output_tagging_flags()
    assert tags["-colorspace"] == "bt709"
    assert tags["-color_primaries"] == "bt709"
    assert tags["-color_trc"] == "bt709"
