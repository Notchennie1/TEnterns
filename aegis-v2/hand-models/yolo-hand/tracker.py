"""
YOLOv8 Hand Tracker (Experimental)
====================================
Uses a YOLOv8-pose model fine-tuned for hand keypoint detection.
This is an experimental alternative to MediaPipe for comparison.

To use:
  1. Train or download a YOLOv8-pose hand model
  2. Place weights at hand-models/yolo-hand/weights/yolo_hand.pt
  3. Set hand_tracker.backend: "yolo-hand" in config
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from hand_models.common import BaseHandTracker, HandDetection, HandLandmark, TrackerRegistry

logger = logging.getLogger("hand_models.yolo_hand")

# Mapping from YOLO hand keypoint indices to landmark names
# Adjust this based on your specific YOLO hand model's keypoint order
_YOLO_HAND_KEYPOINTS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class YOLOHandTracker(BaseHandTracker):
    """Hand tracker using YOLOv8-pose fine-tuned for hands."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._model_path = config.get(
            "model_path",
            str(Path(__file__).parent / "weights" / "yolo_hand.pt"),
        )
        self._device = config.get("device", "cpu")
        self._model = None

    @property
    def name(self) -> str:
        return "YOLOv8 Hand Pose"

    def load_model(self) -> None:
        from ultralytics import YOLO

        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"YOLO hand model not found: {self._model_path}\n"
                "Train one or download a pre-trained hand-pose model."
            )
        logger.info("Loading YOLO hand model: %s", self._model_path)
        self._model = YOLO(self._model_path)

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        detections: list[HandDetection] = []

        for result in results:
            if result.keypoints is None:
                continue

            for idx in range(len(result.keypoints)):
                kps = result.keypoints[idx].data[0].cpu().numpy()  # (N, 3): x, y, conf
                box = result.boxes[idx].xyxy[0].cpu().numpy() if result.boxes else None

                landmarks = []
                for kp_idx, kp in enumerate(kps):
                    if kp_idx < len(_YOLO_HAND_KEYPOINTS):
                        landmarks.append(HandLandmark(
                            name=_YOLO_HAND_KEYPOINTS[kp_idx],
                            x=float(kp[0]),
                            y=float(kp[1]),
                            z=0.0,
                            confidence=float(kp[2]) if len(kp) > 2 else 1.0,
                        ))

                bbox = tuple(box.astype(float)) if box is not None else None

                detections.append(HandDetection(
                    hand_id=idx,
                    handedness="unknown",  # YOLO doesn't classify handedness by default
                    landmarks=landmarks,
                    bounding_box=bbox,
                ))

        return detections

    def release(self) -> None:
        self._model = None
        logger.info("YOLO hand model released")


# Auto-register
TrackerRegistry.register("yolo-hand", YOLOHandTracker)
