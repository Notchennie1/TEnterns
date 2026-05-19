from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
from collections import deque
from pathlib import Path
import math
import time

import cv2
import mediapipe as mp

# Google AI Edge / MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class HandLandmarkResult:
    handedness: str
    landmarks: List[Any]
    is_grabbing: bool
    grab_score: float


class HandTracker:
    """Hand tracker using Google AI Edge (MediaPipe Tasks) Hand Landmarker.

    Notes:
    - This uses Tasks API (not legacy mp.solutions) and expects a `.task` model.
    - `detect(frame_bgr)` returns the same structure your pipeline already uses.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        grab_threshold_ratio: float = 1.15,
        grab_confirm_frames: int = 3,
        release_confirm_frames: int = 2,
        min_handedness_score: float = 0.0,
        model_path: str | None = None,
    ) -> None:
        # `static_image_mode` is a legacy Solutions API concept; Tasks uses explicit running modes.
        # We always use VIDEO mode for webcam-style pipelines (enables tracking between frames).
        _ = static_image_mode

        repo_root = Path(__file__).resolve().parents[3]
        default_model = repo_root / "models" / "hand_landmarker.task"
        model_asset_path = Path(model_path) if model_path else default_model

        if not model_asset_path.exists():
            raise FileNotFoundError(
                "Hand Landmarker model not found. Download it to:\n"
                f"  {model_asset_path}\n\n"
                "Recommended model (float16):\n"
                "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_asset_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        # GRAB/OPEN heuristic (ported from the original repo):
        # - Compute palm center (wrist + MCP joints)
        # - For each fingertip (thumb/index/middle/ring/pinky), compute:
        #     normalized_distance = ||tip - palm_center|| / ||wrist - middle_mcp||
        # - Count fingertips with normalized_distance < threshold
        # - GRAB if >= 4 fingers are "curled"
        # NOTE: name kept for backward-compat, but this is a *distance threshold*, not a pinch ratio.
        self._grab_threshold_ratio = float(grab_threshold_ratio)
        self._grab_confirm_frames = max(int(grab_confirm_frames), 1)
        self._release_confirm_frames = max(int(release_confirm_frames), 1)
        self._min_handedness_score = float(min_handedness_score)

        self._grab_history: dict[str, deque[bool]] = {}
        self._stable_grab: dict[str, bool] = {}

        self._last_timestamp_ms = 0

    @staticmethod
    def _distance_2d(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def _distance_3d(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _timestamp_ms(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def _estimate_grab_status(self, landmarks: List[Any]) -> tuple[bool, float]:
        """Return (raw_is_grabbing, grab_score) using normalized fingertip curl distances.

        This mirrors the original repo's logic (closed-fist proxy): fingertips closer to the
        palm center implies fingers are curled.
        """

        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_size = max(self._distance_3d(wrist, middle_mcp), 1e-6)

        # Palm center = average of wrist + MCPs (index, middle, ring, pinky).
        palm_ids = [0, 5, 9, 13, 17]
        cx = sum(landmarks[i].x for i in palm_ids) / len(palm_ids)
        cy = sum(landmarks[i].y for i in palm_ids) / len(palm_ids)
        cz = sum(landmarks[i].z for i in palm_ids) / len(palm_ids)

        class _Point:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        palm_center = _Point(cx, cy, cz)

        tip_ids = [4, 8, 12, 16, 20]
        normalized_tip_distances = [self._distance_3d(landmarks[idx], palm_center) / palm_size for idx in tip_ids]

        curled_count = sum(distance < self._grab_threshold_ratio for distance in normalized_tip_distances)
        grab_score = curled_count / len(tip_ids)
        raw_is_grabbing = curled_count >= 4
        return raw_is_grabbing, grab_score

    def _debounce_grab(self, hand_key: str, raw_is_grabbing: bool) -> bool:
        history = self._grab_history.get(hand_key)
        if history is None:
            history = deque(maxlen=max(self._grab_confirm_frames, self._release_confirm_frames))
            self._grab_history[hand_key] = history

        history.append(raw_is_grabbing)
        stable = self._stable_grab.get(hand_key, False)

        if len(history) >= self._grab_confirm_frames and all(list(history)[-self._grab_confirm_frames :]):
            stable = True
        elif (
            len(history) >= self._release_confirm_frames
            and not any(list(history)[-self._release_confirm_frames :])
        ):
            stable = False

        self._stable_grab[hand_key] = stable
        return stable

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        detections: List[HandLandmarkResult] = []

        hand_landmarks_list = getattr(results, "hand_landmarks", None) or []
        world_landmarks_list = getattr(results, "hand_world_landmarks", None) or []
        handedness_list = getattr(results, "handedness", None) or []

        for idx, (hand_landmarks, hand_handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
            # Keep normalized landmarks for drawing (x/y in [0,1]).
            landmarks = list(hand_landmarks)

            # Prefer world landmarks for grab computation when available (more stable across camera distance).
            grab_landmarks = (
                list(world_landmarks_list[idx])
                if idx < len(world_landmarks_list) and world_landmarks_list[idx]
                else landmarks
            )

            category = hand_handedness[0] if hand_handedness else None
            handedness = (
                getattr(category, "category_name", None)
                or getattr(category, "display_name", None)
                or "Unknown"
            )
            handedness_score = float(getattr(category, "score", 0.0)) if category else 0.0
            if handedness_score < self._min_handedness_score:
                continue

            raw_is_grabbing, grab_score = self._estimate_grab_status(grab_landmarks)
            is_grabbing = self._debounce_grab(handedness, raw_is_grabbing)
            detections.append(
                HandLandmarkResult(
                    handedness=handedness,
                    landmarks=landmarks,
                    is_grabbing=is_grabbing,
                    grab_score=grab_score,
                )
            )

        return detections

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
