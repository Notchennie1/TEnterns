"""
Pipeline Orchestrator
=====================
Ties together the three stages of the system:

  1. INITIALIZATION  — Capture a snapshot, run the bin boundary detector,
                       lock the bin coordinates for the session.
  2. TRACKING LOOP   — Continuously read frames, run the hand tracker,
                       and feed detections into the bin assignment engine.
  3. OUTPUT / UI     — Render an overlay showing bin outlines, hand
                       landmarks, and which bin is currently active.

This is the single entry point that wires every module together.
"""

from __future__ import annotations

import logging
import time

import cv2

from bin_tracker.detectors import BinBoundaryDetector
from bin_tracker.engine import BinAssignmentEngine
from bin_tracker.trackers import TrackerRegistry
from bin_tracker.trackers.base_hand_tracker import BaseHandTracker
from bin_tracker.ui import OverlayUI
from bin_tracker.utils import load_config, setup_logger

# Ensure built-in backends are registered
import bin_tracker.trackers.mediapipe_tracker  # noqa: F401

logger = logging.getLogger("bin_tracker.pipeline")


class Pipeline:
    """
    Main orchestrator — call run() to start the full lifecycle.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self._config = load_config(config_path)
        setup_logger(config=self._config)

        # Sub-components (created during init phase)
        self._cap: cv2.VideoCapture | None = None
        self._bin_detector: BinBoundaryDetector | None = None
        self._hand_tracker: BaseHandTracker | None = None
        self._engine: BinAssignmentEngine | None = None
        self._ui: OverlayUI | None = None

    # ── Lifecycle ────────────────────────────────────────────

    def run(self) -> None:
        """Full lifecycle: initialize → loop → cleanup."""
        try:
            self._open_camera()
            self._initialize_bins()
            self._create_hand_tracker()
            self._create_engine()
            self._create_ui()
            self._tracking_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    # ── Stage 1: Initialization ──────────────────────────────

    def _open_camera(self) -> None:
        cam_cfg = self._config["camera"]
        source = cam_cfg["source"]
        logger.info("Opening camera source: %s", source)

        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        self._cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

    def _initialize_bins(self) -> None:
        """Snapshot → detect bin boundaries → lock coordinates."""
        cfg = self._config["bin_detector"]
        self._bin_detector = BinBoundaryDetector(cfg)

        if cfg.get("snapshot_on_startup", True):
            logger.info("Taking initialization snapshot...")
            self._bin_regions = self._bin_detector.initialize_from_camera(self._cap)
            logger.info("Bin map locked: %d regions", len(self._bin_regions))
        else:
            self._bin_regions = []
            logger.warning("Snapshot disabled — no bin boundaries loaded")

    def _create_hand_tracker(self) -> None:
        ht_cfg = self._config["hand_tracker"]
        backend_name = ht_cfg["backend"]
        logger.info("Creating hand tracker: %s", backend_name)
        self._hand_tracker = TrackerRegistry.create(backend_name, ht_cfg)

    def _create_engine(self) -> None:
        self._engine = BinAssignmentEngine(self._config["bin_assignment"])
        self._engine.set_bin_map(self._bin_regions)

    def _create_ui(self) -> None:
        ui_cfg = self._config["ui"]
        if ui_cfg.get("enabled", True):
            self._ui = OverlayUI(ui_cfg, self._bin_regions)

    # ── Stage 2: Real-time tracking loop ─────────────────────

    def _tracking_loop(self) -> None:
        logger.info("Entering tracking loop (press 'q' to quit)...")
        frame_count = 0
        t0 = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame — retrying")
                continue

            # Detect hands
            hands = self._hand_tracker.detect(frame)

            # Assign hands to bins
            events = self._engine.assign(hands)

            # Log events
            for ev in events:
                if ev.bin_id is not None:
                    logger.debug("Hand %d (%s) → %s", ev.hand_id, ev.handedness, ev.bin_label)

            # Render UI overlay
            if self._ui:
                display = self._ui.render(frame, hands, events)
                cv2.imshow(self._config["ui"].get("window_name", "Bin Tracker"), display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed")
                    break

            frame_count += 1
            if frame_count % 300 == 0:
                fps = frame_count / (time.time() - t0)
                logger.info("FPS: %.1f", fps)

    # ── Cleanup ──────────────────────────────────────────────

    def _cleanup(self) -> None:
        logger.info("Shutting down...")
        if self._hand_tracker:
            self._hand_tracker.release()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Done.")
