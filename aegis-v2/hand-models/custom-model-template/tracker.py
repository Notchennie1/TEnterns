"""
Custom Hand Tracker — Template
===============================
Copy this folder to create a new hand tracking backend.

Steps:
  1. Copy this folder: cp -r custom-model-template/ my-new-model/
  2. Rename the class and implement load_model() + detect()
  3. Update the TrackerRegistry.register() call at the bottom
  4. Add any model weights to my-new-model/weights/
  5. Test with: python -m hand_models.benchmarks.run_benchmark --backend my-backend-name

The integration pipeline will pick it up automatically when you set:
    hand_tracker.backend: "my-backend-name"  (in config)
"""

from __future__ import annotations

import logging

import numpy as np

from hand_models.common import BaseHandTracker, HandDetection, HandLandmark, TrackerRegistry

logger = logging.getLogger("hand_models.custom")


class CustomHandTracker(BaseHandTracker):
    """
    Template hand tracker — replace with your implementation.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Add your model-specific config here
        self._model = None

    @property
    def name(self) -> str:
        return "Custom Hand Tracker"

    def load_model(self) -> None:
        """
        Load your model here.

        Example:
            import onnxruntime as ort
            self._model = ort.InferenceSession("weights/my_model.onnx")
        """
        logger.info("Loading custom hand model...")
        # TODO: Replace with your model loading code
        raise NotImplementedError("Implement load_model() for your custom tracker")

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        """
        Run inference on a single BGR frame and return detections.

        Your implementation should:
          1. Preprocess the frame (resize, normalize, etc.)
          2. Run your model
          3. Post-process outputs into HandDetection objects
          4. Return the list of detections

        Each HandDetection should have:
          - hand_id: int
          - handedness: "left" | "right" | "unknown"
          - landmarks: list[HandLandmark]  (at minimum: wrist, index_tip)
          - bounding_box: (x1, y1, x2, y2) in pixels
          - is_grabbing: bool (optional)
          - grab_score: float (optional)
        """
        raise NotImplementedError("Implement detect() for your custom tracker")

    def release(self) -> None:
        """Free resources (GPU memory, file handles, etc.)."""
        self._model = None


# Register your backend — change "custom-template" to your chosen name
# TrackerRegistry.register("my-backend-name", CustomHandTracker)
