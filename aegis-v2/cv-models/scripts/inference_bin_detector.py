"""
Bin Boundary Inference Module
=============================
Standalone inference script for the trained YOLOv8-seg bin detector.
Used by the integration pipeline at startup to snapshot the bin layout.

Usage:
    # Test on a single image
    python inference_bin_detector.py --model weights/best.pt --source test.jpg

    # Test on webcam (takes one snapshot)
    python inference_bin_detector.py --model weights/best.pt --source 0 --snapshot
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BinRegion:
    """One detected bin with its segmentation polygon."""
    bin_id: int
    label: str
    polygon: np.ndarray          # shape (N, 2) — x, y vertices in pixel coords
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.polygon is not None and len(self.polygon) > 0:
            self.centroid = (
                float(self.polygon[:, 0].mean()),
                float(self.polygon[:, 1].mean()),
            )


class BinBoundaryDetector:
    """
    Loads a trained YOLOv8-seg model and detects bin boundaries.

    Usage:
        detector = BinBoundaryDetector("weights/best.pt")
        bins = detector.detect(frame)   # list[BinRegion]
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self._model_path = model_path
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._device = device
        self._model = None

    def load_model(self) -> None:
        from ultralytics import YOLO
        logger.info("Loading bin segmentation model: %s", self._model_path)
        self._model = YOLO(self._model_path)

    def detect(self, frame: np.ndarray) -> list[BinRegion]:
        """Run segmentation on a single frame and return detected bins."""
        if self._model is None:
            self.load_model()

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_from_camera(self, cap: cv2.VideoCapture) -> list[BinRegion]:
        """Convenience: grab one frame and detect."""
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return self.detect(frame)

    def _parse_results(self, results) -> list[BinRegion]:
        bins: list[BinRegion] = []

        for result in results:
            # Prefer segmentation masks (polygon outlines)
            if result.masks is not None:
                for idx, mask_xy in enumerate(result.masks.xy):
                    polygon = np.array(mask_xy, dtype=np.float32)
                    conf = float(result.boxes.conf[idx]) if result.boxes is not None else 0.0
                    box = result.boxes.xyxy[idx].cpu().numpy().astype(int)
                    bins.append(BinRegion(
                        bin_id=idx,
                        label=f"bin_{idx}",
                        polygon=polygon,
                        bbox=tuple(box),
                        confidence=conf,
                    ))
            # Fallback to bounding boxes
            elif result.boxes is not None:
                for idx, box in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    polygon = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
                    ], dtype=np.float32)
                    conf = float(result.boxes.conf[idx])
                    bins.append(BinRegion(
                        bin_id=idx,
                        label=f"bin_{idx}",
                        polygon=polygon,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                    ))

        # Sort spatially: top-to-bottom, then left-to-right
        bins.sort(key=lambda b: (b.centroid[1], b.centroid[0]))
        for i, b in enumerate(bins):
            b.bin_id = i
            b.label = f"bin_{i}"

        return bins


def visualize(frame: np.ndarray, bins: list[BinRegion]) -> np.ndarray:
    """Draw detected bin boundaries on frame."""
    vis = frame.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]
    for b in bins:
        color = colors[b.bin_id % len(colors)]
        pts = b.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(
            vis, f"{b.label} ({b.confidence:.2f})",
            (int(b.centroid[0]) - 30, int(b.centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Bin boundary inference")
    parser.add_argument("--model", required=True, help="Path to trained .pt weights")
    parser.add_argument("--source", default="0", help="Image path or camera index")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--snapshot", action="store_true", help="Take single snapshot from camera")
    parser.add_argument("--save", default=None, help="Save visualization to this path")
    args = parser.parse_args()

    detector = BinBoundaryDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )

    # Determine source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    if isinstance(source, int) or args.snapshot:
        cap = cv2.VideoCapture(source if isinstance(source, int) else 0)
        bins = detector.detect_from_camera(cap)
        ret, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        bins = detector.detect(frame)

    print(f"\nDetected {len(bins)} bin(s):")
    for b in bins:
        print(f"  {b.label}: centroid=({b.centroid[0]:.0f}, {b.centroid[1]:.0f}), "
              f"conf={b.confidence:.2f}, points={len(b.polygon)}")

    vis = visualize(frame, bins)
    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Saved to {args.save}")
    else:
        cv2.imshow("Bin Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
