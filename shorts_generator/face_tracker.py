"""Offline computer vision face detection and speaker tracking engine.

Integrates OpenCV Haar cascades to detect facial landmarks and centroids
across temporal video windows with zero cloud dependencies. Formats spatial
coordinates for deadband filtering and hysteresis smoothing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class FaceTracker:
    """Offline computer vision face detector and active speaker tracker."""

    def __init__(self, cascade_path: Optional[str] = None) -> None:
        """Initialize FaceTracker with Haar cascade classifier.

        Args:
            cascade_path: Optional custom path to cascade XML. Defaults to OpenCV's built-in frontal face cascade.
        """
        if not cascade_path:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade_path = cascade_path
        self._classifier: Optional[cv2.CascadeClassifier] = None

    @property
    def classifier(self) -> cv2.CascadeClassifier:
        """Lazy-load the cascade classifier instance.

        Returns:
            Configured cv2.CascadeClassifier.
        """
        if self._classifier is None:
            self._classifier = cv2.CascadeClassifier(self.cascade_path)
        return self._classifier

    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular face bounding boxes in a single BGR/RGB frame.

        Args:
            frame: Numpy image array (BGR or RGB).

        Returns:
            List of (x, y, w, h) bounding box coordinates.
        """
        if frame is None or frame.size == 0:
            return []

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def sample_video_faces(
        self,
        media_path: str,
        start_time: float,
        end_time: float,
        fps_sample: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Sample video frames across highlight window and track speaker centroids.

        Args:
            media_path: Path to local video file.
            start_time: Clip start timestamp in seconds.
            end_time: Clip end timestamp in seconds.
            fps_sample: Sampling frequency in frames per second.

        Returns:
            List of detection dictionaries with 'timestamp', 'center_x', 'face_box', and 'speaker_id'.
        """
        if not os.path.exists(media_path):
            return []

        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            return []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        step_sec = max(0.2, 1.0 / max(0.5, fps_sample))
        detections: List[Dict[str, Any]] = []

        curr_t = start_time
        while curr_t <= end_time:
            cap.set(cv2.CAP_PROP_POS_MSEC, curr_t * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            boxes = self.detect_faces_in_frame(frame)
            if boxes:
                boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
                primary_box = boxes[0]
                center_x = primary_box[0] + (primary_box[2] / 2.0)
                rel_t = max(0.0, curr_t - start_time)
                speaker_id = 0 if center_x < (frame_width * 0.5) else 1

                detections.append({
                    "timestamp": round(rel_t, 3),
                    "center_x": round(center_x, 2),
                    "face_box": primary_box,
                    "speaker_id": speaker_id,
                })

            curr_t += step_sec

        cap.release()
        return detections
