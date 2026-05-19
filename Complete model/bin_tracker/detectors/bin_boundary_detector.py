"""
Bin Boundary Detector
=====================
Uses a pre-trained YOLO segmentation model to detect bin boundaries from a
single snapshot frame.  Because the camera is fixed, the detected polygons
are captured once at startup and reused for every subsequent frame.

Flow:
  1. Load YOLO .pt weights.
  2. Capture one snapshot from the camera.
  3. Run inference → extract segmentation masks / bounding polygons.
  4. Store the polygons as the fixed "bin map" for the session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("bin_tracker.detectors")


# ── Data structures ──────────────────────────────────────────

@dataclass
class BinRegion:
    """One detected bin and its fixed boundary polygon."""
    bin_id: int
    label: str                              # e.g. "bin_0", "bin_1"
    polygon: np.ndarray                     # shape (N, 2) — x, y vertices
    centroid: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.polygon is not None and len(self.polygon) > 0:
            self.centroid = (
                float(self.polygon[:, 0].mean()),
                float(self.polygon[:, 1].mean()),
            )


# ── Detector ─────────────────────────────────────────────────

class BinBoundaryDetector:
    """
    Plug-and-play bin boundary initializer.

    Usage:
        detector = BinBoundaryDetector(config["bin_detector"])
        bins = detector.initialize(frame)   # run once at startup
        # bins is a list[BinRegion] that stays fixed for the session
    """

    def __init__(self, config: dict):
        self._model_path: str = config["model_path"]
        self._conf: float = config.get("confidence_threshold", 0.5)
        self._iou: float = config.get("iou_threshold", 0.45)
        self._device: str = config.get("device", "cpu")
        self._model = None

    # ── Public API ───────────────────────────────────────────

    def load_model(self) -> None:
        """Load the YOLO segmentation model from disk."""
        from ultralytics import YOLO

        logger.info("Loading bin segmentation model from %s", self._model_path)
        self._model = YOLO(self._model_path)
        logger.info("Model loaded on device=%s", self._device)

    def initialize(self, frame: np.ndarray) -> list[BinRegion]:
        """
        Run a one-shot detection on *frame* and return the fixed bin map.

        Parameters
        ----------
        frame : np.ndarray
            The initialization snapshot (BGR image from the camera).

        Returns
        -------
        list[BinRegion]
            Detected bin regions with polygon boundaries.
        """
        if self._model is None:
            self.load_model()

        logger.info("Running bin boundary detection on snapshot (%dx%d)",
                     frame.shape[1], frame.shape[0])

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        bins = self._parse_results(results)
        logger.info("Detected %d bin region(s)", len(bins))
        return bins

    def initialize_from_camera(self, cap: cv2.VideoCapture) -> list[BinRegion]:
        """Convenience: grab one frame from *cap* and run initialization."""
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture snapshot from camera")
        return self.initialize(frame)

    # ── Internals ────────────────────────────────────────────

    def _parse_results(self, results) -> list[BinRegion]:
        """Convert YOLO results into a list of BinRegion objects."""
        bins: list[BinRegion] = []

        for result in results:
            # If model returns segmentation masks
            if result.masks is not None:
                for idx, mask in enumerate(result.masks.xy):
                    polygon = np.array(mask, dtype=np.float32)
                    conf = float(result.boxes.conf[idx]) if result.boxes is not None else 0.0
                    bins.append(BinRegion(
                        bin_id=idx,
                        label=f"bin_{idx}",
                        polygon=polygon,
                        confidence=conf,
                    ))

            # Fallback: use bounding boxes as rectangular polygons
            elif result.boxes is not None:
                for idx, box in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    polygon = np.array([
                        [x1, y1], [x2, y1],
                        [x2, y2], [x1, y2],
                    ], dtype=np.float32)
                    conf = float(result.boxes.conf[idx])
                    bins.append(BinRegion(
                        bin_id=idx,
                        label=f"bin_{idx}",
                        polygon=polygon,
                        confidence=conf,
                    ))

        return bins
