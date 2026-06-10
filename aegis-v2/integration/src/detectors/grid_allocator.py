"""
Grid Allocator
==============
Pure functions that place raw OBB bin detections onto a FIXED, pre-indexed grid.
No camera / model / cv2 — just geometry, so it unit-tests in milliseconds.

Contract
--------
* The grid is indexed FIRST and is fixed. Indices come from the grid position,
  never from detection order.
* A detection is dropped into the cell it occupies; a cell with no detection stays
  ``detected=False`` (no renumbering of any other index).
* Bin spans are hardcoded by row (top = 1 cell, bottom = 2 cells). Inferring span /
  size from box geometry, and generalising beyond the fixed grid, are FUTURE_TASKS.md.

Output: dict keyed ``bin_{index}`` (index is the canonical 1..N bin id).
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger("aegis.detectors.grid_allocator")

# Fixed rig: top row of 6 single-cell bins, bottom row of 3 double-cell bins.
DEFAULT_LAYOUT = [[1, 1, 1, 1, 1, 1], [2, 2, 2]]


def _layer_name(row: int, num_rows: int) -> str:
    if num_rows == 2:
        return "top" if row == 0 else "bottom"
    return f"row{row}"


def build_skeleton(layout: list[list[int]]) -> dict:
    """Fixed, pre-indexed grid. Index runs 1..N across rows (top row first)."""
    skeleton: dict = {}
    num_rows = len(layout)
    index = 1
    for row, spans in enumerate(layout):
        row_slots = sum(int(s) for s in spans)
        num_bins = len(spans)
        slot_start = 0
        for col, span in enumerate(spans):
            span = int(span)
            skeleton[f"bin_{index}"] = {
                "index": index,
                "layer": _layer_name(row, num_rows),
                "row": row,
                "col": col,
                "slot_start": slot_start,
                "span": span,
                "row_slots": row_slots,
                "num_bins": num_bins,
                "detected": False,
                "confidence": 0.0,
            }
            slot_start += span
            index += 1
    return skeleton


def _cy(det) -> float:
    return float(det["center"][1])


def _cx(det) -> float:
    return float(det["center"][0])


def _box_height(det) -> float:
    ys = [p[1] for p in det.get("corners", [])]
    return float(max(ys) - min(ys)) if ys else 0.0


def _median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def split_rows_by_y(detections: list, num_rows: int) -> list:
    """Group detections into ``num_rows`` rows by splitting at SIGNIFICANT y-gaps.

    A gap counts as a row boundary only if it exceeds ~half the median bin height,
    so detections that share a row (near-equal y) are never force-split. At most
    ``num_rows - 1`` boundaries are taken (the largest qualifying gaps). When fewer
    qualify, later rows stay empty and earlier rows absorb the detections (a single
    detected band lands in row 0). Robust when fewer detections than rows are present.
    """
    rows = [[] for _ in range(num_rows)]
    if not detections:
        return rows
    ordered = sorted(detections, key=_cy)
    if num_rows == 1 or len(ordered) == 1:
        rows[0] = list(ordered)
        return rows

    threshold = 0.5 * _median([_box_height(d) for d in ordered])
    # gaps[i] = vertical distance between ordered[i] and ordered[i+1]
    gaps = [(ordered[i + 1]["center"][1] - ordered[i]["center"][1], i)
            for i in range(len(ordered) - 1)]
    significant = [(g, i) for g, i in gaps if g > threshold]
    # largest qualifying gaps first (sort by gap value only — never by index)
    significant.sort(key=lambda t: t[0], reverse=True)
    boundaries = sorted(i for _, i in significant[:num_rows - 1])

    band = 0
    start = 0
    for b in boundaries:
        rows[band] = ordered[start:b + 1]
        band += 1
        start = b + 1
    rows[band] = ordered[start:]
    return rows
