"""
MediaPipe Hands — example backend implementation.
==================================================
This serves as the reference plug-in.  Swap it out by changing
`hand_tracker.backend` in config.yaml to any other registered name.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .base_hand_tracker import BaseHandTracker, HandDetection, HandLandmark
from .registry import TrackerRegistry

logger = logging.getLogger("bin_tracker.trackers.mediapipe")

# MediaPipe hand landmark names (21 keypoints)
_MP_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class MediaPipeHandTracker(BaseHandTracker):
    """Hand tracker using Google MediaPipe Hands."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._hands = None

    def load_model(self) -> None:
        import mediapipe as mp

        logger.info("Initializing MediaPipe Hands (max_hands=%d)", self._max_hands)
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max_hands,
            min_detection_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
        )

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        if self._hands is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        detections: list[HandDetection] = []

        if not result.multi_hand_landmarks:
            return detections

        for idx, hand_lms in enumerate(result.multi_hand_landmarks):
            # Determine handedness
            handedness = "unknown"
            if result.multi_handedness and idx < len(result.multi_handedness):
                handedness = result.multi_handedness[idx].classification[0].label.lower()

            landmarks = []
            for lm_idx, lm in enumerate(hand_lms.landmark):
                landmarks.append(HandLandmark(
                    name=_MP_LANDMARK_NAMES[lm_idx],
                    x=lm.x * w,
                    y=lm.y * h,
                    z=lm.z,
                    confidence=lm.visibility if hasattr(lm, "visibility") else 1.0,
                ))

            # Bounding box from landmark extents
            xs = [l.x for l in landmarks]
            ys = [l.y for l in landmarks]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            detections.append(HandDetection(
                hand_id=idx,
                handedness=handedness,
                landmarks=landmarks,
                bounding_box=bbox,
            ))

        return detections

    def release(self) -> None:
        if self._hands:
            self._hands.close()
            logger.info("MediaPipe Hands released")


# ── Auto-register on import ─────────────────────────────────
TrackerRegistry.register("mediapipe", MediaPipeHandTracker)
