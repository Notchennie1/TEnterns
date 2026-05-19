"""
Shared Pipeline State
======================
Thread-safe state object that the pipeline loop writes to and the
FastAPI dashboard reads from. This is the bridge between the real-time
CV pipeline and the web UI.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BinStatus:
    """Status of a single bin for the dashboard."""
    bin_id: str
    label: str
    x_min: float = 0
    x_max: float = 0
    y_min: float = 0
    y_max: float = 0
    confidence: float = 0.0
    is_active: bool = False          # Hand currently in this bin
    hand_id: Optional[int] = None
    handedness: str = ""
    pick_count: int = 0              # Successful picks from this bin
    target_count: int = 0            # Expected picks (from work order)
    using: bool = True               # Is this bin part of the current job


@dataclass
class HandStatus:
    """Current status of a detected hand."""
    hand_id: int
    handedness: str
    position: tuple[float, float] = (0.0, 0.0)
    is_grabbing: bool = False
    grab_score: float = 0.0
    assigned_bin: Optional[str] = None


@dataclass
class PipelineState:
    """
    Thread-safe container for the live pipeline state.

    The pipeline loop calls update_*() methods.
    The FastAPI endpoints call get_*() methods.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._bins: dict[str, BinStatus] = {}
        self._hands: list[HandStatus] = []
        self._fsm_state: str = "idle"
        self._fsm_bin_id: Optional[str] = None
        self._fsm_elapsed: float = 0.0
        self._errors: list[dict] = []
        self._fps: float = 0.0
        self._last_update: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = time.time()

    # ── Writers (called by the pipeline loop) ────────────────

    def update_bins(self, geofences: dict, active_bin_ids: set[str] | None = None) -> None:
        """Update bin map from detected geofences."""
        with self._lock:
            active = active_bin_ids or set()
            for bid, coords in geofences.items():
                if bid not in self._bins:
                    self._bins[bid] = BinStatus(bin_id=bid, label=bid)
                b = self._bins[bid]
                b.x_min = coords.get("x_min", 0)
                b.x_max = coords.get("x_max", 0)
                b.y_min = coords.get("y_min", 0)
                b.y_max = coords.get("y_max", 0)
                b.confidence = coords.get("confidence", 0.0)
                b.is_active = bid in active

    def update_hands(self, hands: list, events: list) -> None:
        """Update hand positions and bin assignments."""
        with self._lock:
            self._hands = []
            event_map = {ev.hand_id: ev for ev in events}
            for hand in hands:
                center = getattr(hand, "center", None)
                if center is None:
                    center = (0.0, 0.0)
                ev = event_map.get(hand.hand_id)
                self._hands.append(HandStatus(
                    hand_id=hand.hand_id,
                    handedness=hand.handedness,
                    position=center,
                    is_grabbing=getattr(hand, "is_grabbing", False),
                    grab_score=getattr(hand, "grab_score", 0.0),
                    assigned_bin=ev.bin_id if ev else None,
                ))

            # Update bin active states
            active_ids = {ev.bin_id for ev in events if ev.bin_id is not None}
            for bid, b in self._bins.items():
                b.is_active = bid in active_ids
                matching = [ev for ev in events if ev.bin_id == bid]
                if matching:
                    b.hand_id = matching[0].hand_id
                    b.handedness = matching[0].handedness
                else:
                    b.hand_id = None
                    b.handedness = ""

    def update_fsm(self, state: str, bin_id: Optional[str], elapsed: float) -> None:
        """Update FSM state display."""
        with self._lock:
            self._fsm_state = state
            self._fsm_bin_id = bin_id
            self._fsm_elapsed = elapsed

    def record_pick(self, bin_id: str) -> None:
        """Increment pick count for a bin (called on FSM success)."""
        with self._lock:
            if bin_id in self._bins:
                self._bins[bin_id].pick_count += 1

    def add_error(self, bin_id: str, message: str) -> None:
        """Record an error event."""
        with self._lock:
            self._errors.append({
                "bin_id": bin_id,
                "message": message,
                "timestamp": time.time(),
            })
            # Keep last 50 errors
            if len(self._errors) > 50:
                self._errors = self._errors[-50:]

    def clear_errors(self) -> None:
        with self._lock:
            self._errors.clear()

    def update_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = fps
            self._last_update = time.time()
            self._frame_count += 1

    # ── Readers (called by FastAPI endpoints) ────────────────

    def get_bins(self) -> list[dict]:
        with self._lock:
            result = []
            for b in self._bins.values():
                status = self._calculate_bin_status(b)
                result.append({
                    "id": b.bin_id,
                    "label": b.label,
                    "current": b.pick_count,
                    "total": b.target_count,
                    "status": status,
                    "using": b.using,
                    "is_active": b.is_active,
                    "handedness": b.handedness,
                    "confidence": b.confidence,
                })
            return result

    def get_hands(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "hand_id": h.hand_id,
                    "handedness": h.handedness,
                    "x": h.position[0],
                    "y": h.position[1],
                    "is_grabbing": h.is_grabbing,
                    "grab_score": h.grab_score,
                    "assigned_bin": h.assigned_bin,
                }
                for h in self._hands
            ]

    def get_fsm(self) -> dict:
        with self._lock:
            return {
                "state": self._fsm_state,
                "bin_id": self._fsm_bin_id,
                "elapsed": self._fsm_elapsed,
            }

    def get_errors(self) -> list[dict]:
        with self._lock:
            return list(self._errors)

    def get_stats(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "fps": self._fps,
                "frame_count": self._frame_count,
                "uptime_seconds": uptime,
                "last_update": self._last_update,
                "num_bins": len(self._bins),
                "num_hands": len(self._hands),
            }

    def set_work_order(self, bin_targets: dict[str, int]) -> None:
        """Set target pick counts per bin from a work order."""
        with self._lock:
            for bid, target in bin_targets.items():
                if bid in self._bins:
                    self._bins[bid].target_count = target
                    self._bins[bid].using = target > 0

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _calculate_bin_status(b: BinStatus) -> str:
        """Determine display status for a bin."""
        if not b.using:
            if b.is_active:
                return "wrong_bin"
            return "grey"

        if b.is_active:
            return "active"

        if b.target_count == 0:
            return "white"
        elif b.pick_count == 0:
            return "white"
        elif b.pick_count < b.target_count:
            return "orange"
        elif b.pick_count == b.target_count:
            return "green"
        else:
            return "warn"
