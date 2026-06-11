"""
Grid Calibrator
===============
Pure functions for the "calibrate from a full snapshot, then match" workflow.
No camera / model / cv2 — just geometry, so it unit-tests in milliseconds.

* ``calibrate_grid`` turns an all-9-bins snapshot into the 9 fixed slots (the grid),
  indexed 1..6 (top) / 7..9 (bottom), storing each slot's real centre + box.
* ``match_to_grid`` matches a later snapshot's detections to the nearest calibrated
  slot (same row, one-to-one, within a distance cutoff) -> per-slot occupancy.

A detection is ``{"corners": [[x,y]*4], "center": [cx, cy], "conf": float}``.
"""
from __future__ import annotations

import logging
import math

import grid_allocator as ga  # reuse the proven row split

logger = logging.getLogger("aegis.detectors.grid_calibrator")


def _cx(det) -> float:
    return float(det["center"][0])


def _cy(det) -> float:
    return float(det["center"][1])


def _median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def calibrate_grid(detections: list) -> dict:
    """All-9-bins snapshot -> the 9 fixed slots (the grid).

    Splits into 2 rows by y-gap, validates exactly 6 top + 3 bottom, orders each
    row left->right, and assigns indices 1..6 (top) / 7..9 (bottom). Stores each
    slot's centre, box, and the row's median centre-to-centre spacing (used as the
    match cutoff later). Raises ValueError if the snapshot isn't a clean 6+3.
    """
    rows = ga.split_rows_by_y(detections, num_rows=2)
    top, bottom = rows[0], rows[1]
    if len(top) != 6 or len(bottom) != 3:
        raise ValueError(
            f"Calibration needs 6 top + 3 bottom bins; got {len(top)} top / "
            f"{len(bottom)} bottom. Retake the calibration snapshot with all 9 bins."
        )

    slots: dict = {}
    index = 1
    for row_idx, layer, span, row_dets in (
        (0, "top", 1, top),
        (1, "bottom", 2, bottom),
    ):
        ordered = sorted(row_dets, key=_cx)
        centers = [_cx(d) for d in ordered]
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        row_spacing = _median(diffs)
        for d in ordered:
            slots[f"slot_{index}"] = {
                "index": index,
                "row": row_idx,
                "layer": layer,
                "span": span,
                "center": [_cx(d), _cy(d)],
                "corners": d["corners"],
                "row_spacing": row_spacing,
            }
            index += 1
    return slots
