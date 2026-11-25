"""Triad unit tests for offline OpenCV face tracking and dynamic pan crop expression compilation."""

from typing import Dict, List
import numpy as np
import pytest

from shorts_generator.face_tracker import FaceTracker
from shorts_generator.framing_analyzer import generate_pan_crop_expression, smooth_speaker_tracking


def test_face_tracker_handles_empty_and_synthetic_frames() -> None:
    """Edge Case & Positive: Verify FaceTracker handles empty, blank, and synthetic frames gracefully."""
    tracker = FaceTracker()

    empty_res = tracker.detect_faces_in_frame(np.array([]))
    assert empty_res == []

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_res = tracker.detect_faces_in_frame(blank_frame)
    assert isinstance(blank_res, list)

    missing_sample = tracker.sample_video_faces("non_existent_media.mp4", 0.0, 5.0)
    assert missing_sample == []


def test_generate_pan_crop_expression_static_and_empty() -> None:
    """Negative: Verify single speaker or empty detections produce stationary integer crop X coordinates."""
    expr_empty = generate_pan_crop_expression([], crop_w=1080, src_w=1920, default_x=420)
    assert expr_empty == "420"

    stationary_detections = [
        {"timestamp": 0.0, "smoothed_x": 960.0},
        {"timestamp": 1.0, "smoothed_x": 960.0},
        {"timestamp": 2.0, "smoothed_x": 960.0},
    ]
    expr_stat = generate_pan_crop_expression(stationary_detections, crop_w=1080, src_w=1920)
    assert expr_stat == "420"


def test_generate_pan_crop_expression_dynamic_transition() -> None:
    """Positive: Verify alternating speaker coordinates produce piecewise linear FFmpeg time expressions."""
    detections: List[Dict] = [
        {"timestamp": 0.0, "smoothed_x": 540.0},
        {"timestamp": 2.5, "smoothed_x": 540.0},
        {"timestamp": 5.0, "smoothed_x": 1380.0},
    ]
    expr = generate_pan_crop_expression(detections, crop_w=1080, src_w=1920)

    assert "if(lte(t," in expr
    assert "min(max(" in expr
    assert ",0),840)" in expr
    assert "t-" in expr


def test_face_tracker_end_to_end_smooth_and_pan() -> None:
    """Positive: Verify seamless pipeline from mock detection sequence to smoothed coordinates and pan expression."""
    raw_detections = [
        {"timestamp": 0.0, "center_x": 400.0, "speaker_id": 0},
        {"timestamp": 0.5, "center_x": 410.0, "speaker_id": 0},
        {"timestamp": 1.0, "center_x": 405.0, "speaker_id": 0},
        {"timestamp": 3.0, "center_x": 1500.0, "speaker_id": 1},
        {"timestamp": 4.0, "center_x": 1500.0, "speaker_id": 1},
        {"timestamp": 5.5, "center_x": 1500.0, "speaker_id": 1},
        {"timestamp": 6.0, "center_x": 1500.0, "speaker_id": 1},
    ]

    smoothed = smooth_speaker_tracking(raw_detections, frame_width=1920, deadband_ratio=0.08, hold_seconds=2.5)
    assert len(smoothed) == len(raw_detections)
    assert smoothed[0]["active_speaker"] == 0
    assert smoothed[-1]["active_speaker"] == 1

    pan_expr = generate_pan_crop_expression(smoothed, crop_w=1080, src_w=1920)
    assert isinstance(pan_expr, str)
    assert "min(max(" in pan_expr
