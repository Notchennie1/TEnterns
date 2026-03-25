from dataclasses import dataclass
from typing import Any, List

import cv2
import mediapipe as mp


@dataclass
class HandLandmarkResult:
    handedness: str
    landmarks: List[Any]


class HandTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        detections: List[HandLandmarkResult] = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                detections.append(
                    HandLandmarkResult(
                        handedness=hand_type.classification[0].label,
                        landmarks=list(hand_landmarks.landmark),
                    )
                )
        return detections
