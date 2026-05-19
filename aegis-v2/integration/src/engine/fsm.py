"""
Triple-Gate FSM (Finite State Machine)
=======================================
Implements the kinetic gating logic:
  Gate 1 — Spatial:  Hand enters a bin geofence
  Gate 2 — Intent:   Grab gesture detected (closed fist)
  Gate 3 — Verify:   Weight change on load receptor confirms pick/place

When all three gates pass, the system activates the load receptor
feedback for that bin.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time
import logging

logger = logging.getLogger("aegis.engine.fsm")


class FSMState(Enum):
    IDLE = "idle"
    GATE_1_SPATIAL = "gate_1_spatial"
    GATE_2_INTENT = "gate_2_intent"
    GATE_3_VERIFY = "gate_3_verify"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class SensorReading:
    """Unified sensor data for one frame."""
    timestamp: float
    hand_in_geofence: bool
    closed_fist_detected: bool
    weight_delta: float             # grams
    bin_id: Optional[str] = None


class TripleGateFSM:
    """
    Triple-Gate Verification State Machine.
    Drives the kinetic gating for load receptor activation.
    """

    def __init__(self, config: dict):
        fsm_cfg = config.get("fsm", config)
        self.state = FSMState.IDLE
        self.previous_state = None
        self.gate_entry_time = None
        self.last_bin_id = None

        self._g1_timeout = fsm_cfg.get("gate1_spatial_timeout", 2.0)
        self._g2_timeout = fsm_cfg.get("gate2_intent_timeout", 1.5)
        self._g3_timeout = fsm_cfg.get("gate3_verification_timeout", 3.0)
        self._weight_threshold = fsm_cfg.get("weight_delta_threshold", 50)
        self._error_cooldown = fsm_cfg.get("error_cooldown", 5.0)

        # Callback for load receptor activation
        self._on_success = None
        self._on_error = None

    def set_callbacks(
        self,
        on_success=None,
        on_error=None,
    ) -> None:
        """Set callbacks for gate pass/fail events."""
        self._on_success = on_success
        self._on_error = on_error

    def update(self, reading: SensorReading) -> FSMState:
        self.previous_state = self.state
        now = time.time()

        if self.state == FSMState.IDLE:
            if reading.hand_in_geofence:
                self.state = FSMState.GATE_1_SPATIAL
                self.gate_entry_time = now
                self.last_bin_id = reading.bin_id
                logger.info("Gate 1 SPATIAL — hand in %s", reading.bin_id)

        elif self.state == FSMState.GATE_1_SPATIAL:
            if (now - self.gate_entry_time) > self._g1_timeout:
                self._fail("Gate 1 timeout")
            elif reading.closed_fist_detected:
                self.state = FSMState.GATE_2_INTENT
                self.gate_entry_time = now
                logger.info("Gate 2 INTENT — grab detected")
            elif not reading.hand_in_geofence:
                self._reset()

        elif self.state == FSMState.GATE_2_INTENT:
            if (now - self.gate_entry_time) > self._g2_timeout:
                self._fail("Gate 2 timeout")
            elif reading.weight_delta >= self._weight_threshold:
                self.state = FSMState.GATE_3_VERIFY
                self.gate_entry_time = now
                logger.info("Gate 3 VERIFY — weight delta: %.1fg", reading.weight_delta)

        elif self.state == FSMState.GATE_3_VERIFY:
            if (now - self.gate_entry_time) > self._g3_timeout:
                self._fail("Gate 3 timeout")
            else:
                self.state = FSMState.SUCCESS
                logger.info("ALL GATES PASSED — bin %s — activating load receptor",
                            self.last_bin_id)
                if self._on_success:
                    self._on_success(self.last_bin_id)

        elif self.state == FSMState.SUCCESS:
            self._reset()

        elif self.state == FSMState.ERROR:
            if (now - self.gate_entry_time) >= self._error_cooldown:
                self._reset()

        return self.state

    def _fail(self, reason: str) -> None:
        self.state = FSMState.ERROR
        self.gate_entry_time = time.time()
        logger.warning("Gate FAILED: %s (was %s)", reason, self.previous_state)
        if self._on_error:
            self._on_error(self.last_bin_id, reason)

    def _reset(self) -> None:
        self.state = FSMState.IDLE
        self.gate_entry_time = None
        self.last_bin_id = None

    def get_state_info(self) -> dict:
        elapsed = time.time() - self.gate_entry_time if self.gate_entry_time else 0
        return {
            "state": self.state.value,
            "elapsed_time": elapsed,
            "bin_id": self.last_bin_id,
            "timestamp": time.time(),
        }
