"""
Bin Assignment Engine
=====================
Given a fixed bin map (list of BinRegions) and real-time hand detections,
determines which bin each hand is currently reaching into.

Supports two strategies:
  • point_in_polygon — precise geometric containment test
  • nearest_centroid  — fallback if hand is between bins
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from bin_tracker.detectors.bin_boundary_detector import BinRegion
from bin_tracker.trackers.base_hand_tracker import HandDetection

logger = logging.getLogger("bin_tracker.engine")


@dataclass
class BinEvent:
    """Represents a hand-in-bin assignment for one frame."""
    hand_id: int
    handedness: str
    bin_id: int | None          # None = hand not inside any bin
    bin_label: str | None
    hand_point: tuple[float, float]
    confidence: float           # from hand tracker
    method: str                 # "point_in_polygon" | "nearest_centroid"


class BinAssignmentEngine:
    """
    Stateless engine: call assign() every frame with the current hand
    detections and the fixed bin map.
    """

    def __init__(self, config: dict):
        self._method: str = config.get("method", "point_in_polygon")
        self._keypoint: str = config.get("hand_keypoint", "index_tip")

        # Pre-built Shapely polygons (set once via set_bin_map)
        self._bin_polygons: list[tuple[BinRegion, Polygon]] = []

    # ── Setup ────────────────────────────────────────────────

    def set_bin_map(self, bins: list[BinRegion]) -> None:
        """
        Lock in the bin boundaries for the session.
        Called once after the initialization snapshot.
        """
        self._bin_polygons = []
        for b in bins:
            poly = Polygon(b.polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)  # auto-fix self-intersections
            self._bin_polygons.append((b, poly))
        logger.info("Bin map set with %d region(s)", len(self._bin_polygons))

    # ── Public API ───────────────────────────────────────────

    def assign(self, hands: list[HandDetection]) -> list[BinEvent]:
        """
        For each detected hand, determine which bin it is reaching into.

        Returns one BinEvent per hand.
        """
        events: list[BinEvent] = []

        for hand in hands:
            point = hand.get_point(self._keypoint)
            if point is None:
                logger.debug("Hand %d missing keypoint '%s', skipping",
                             hand.hand_id, self._keypoint)
                continue

            if self._method == "point_in_polygon":
                event = self._assign_pip(hand, point)
            else:
                event = self._assign_nearest(hand, point)

            events.append(event)

        return events

    # ── Strategies ───────────────────────────────────────────

    def _assign_pip(self, hand: HandDetection, point: tuple[float, float]) -> BinEvent:
        """Point-in-polygon containment test."""
        sp = Point(point)

        for region, poly in self._bin_polygons:
            if poly.contains(sp):
                return BinEvent(
                    hand_id=hand.hand_id,
                    handedness=hand.handedness,
                    bin_id=region.bin_id,
                    bin_label=region.label,
                    hand_point=point,
                    confidence=region.confidence,
                    method="point_in_polygon",
                )

        # Hand not inside any bin
        return BinEvent(
            hand_id=hand.hand_id,
            handedness=hand.handedness,
            bin_id=None,
            bin_label=None,
            hand_point=point,
            confidence=0.0,
            method="point_in_polygon",
        )

    def _assign_nearest(self, hand: HandDetection, point: tuple[float, float]) -> BinEvent:
        """Nearest-centroid fallback."""
        if not self._bin_polygons:
            return BinEvent(
                hand_id=hand.hand_id, handedness=hand.handedness,
                bin_id=None, bin_label=None, hand_point=point,
                confidence=0.0, method="nearest_centroid",
            )

        best_region = None
        best_dist = float("inf")
        px, py = point

        for region, _ in self._bin_polygons:
            cx, cy = region.centroid
            dist = np.hypot(px - cx, py - cy)
            if dist < best_dist:
                best_dist = dist
                best_region = region

        return BinEvent(
            hand_id=hand.hand_id,
            handedness=hand.handedness,
            bin_id=best_region.bin_id,
            bin_label=best_region.label,
            hand_point=point,
            confidence=best_region.confidence,
            method="nearest_centroid",
        )
