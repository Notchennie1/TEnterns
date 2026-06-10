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
