from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BinSegmenter:
    """Placeholder segmenter for bin-boundary masking.

    Replace `segment` implementation with your trained model inference.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None

    def segment(self, frame_bgr: Any) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    @staticmethod
    def extract_bin_boundaries(mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
