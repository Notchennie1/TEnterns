"""Unit tests for grid_allocator — pure geometry, no camera/model/cv2.

Imported by path so the test doesn't drag in the detectors package __init__
(which pulls cv2 / ultralytics).
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import grid_allocator as ga  # noqa: E402

FRAME_W = 1280


def test_skeleton_is_fully_preindexed():
    sk = ga.build_skeleton(ga.DEFAULT_LAYOUT)
    assert len(sk) == 9
    # top row -> indices 1..6, span 1
    assert sk["bin_1"]["index"] == 1 and sk["bin_1"]["span"] == 1
    assert sk["bin_1"]["layer"] == "top" and sk["bin_1"]["slot_start"] == 0
    assert sk["bin_6"]["index"] == 6 and sk["bin_6"]["slot_start"] == 5
    # bottom row -> indices 7..9, span 2, slot_start 0/2/4
    assert sk["bin_7"]["index"] == 7 and sk["bin_7"]["span"] == 2
    assert sk["bin_7"]["layer"] == "bottom" and sk["bin_7"]["slot_start"] == 0
    assert sk["bin_8"]["slot_start"] == 2 and sk["bin_9"]["slot_start"] == 4
    # every cell starts undetected, row_slots 6 for both rows
    assert all(c["detected"] is False for c in sk.values())
    assert sk["bin_1"]["row_slots"] == 6 and sk["bin_9"]["row_slots"] == 6


def make_det(cx, cy, conf=0.9, half=20):
    """A detection dict with a square box centred on (cx, cy)."""
    return {
        "corners": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
        "center": [float(cx), float(cy)],
        "conf": conf,
    }


def test_split_rows_by_largest_y_gap():
    # 6 top dets near y=200, 3 bottom dets near y=560 (big gap between bands)
    top = [make_det(x, 200) for x in (100, 300, 500, 700, 900, 1100)]
    bottom = [make_det(x, 560) for x in (200, 640, 1080)]
    rows = ga.split_rows_by_y(top + bottom, num_rows=2)
    assert len(rows) == 2
    assert len(rows[0]) == 6 and all(d["center"][1] == 200 for d in rows[0])
    assert len(rows[1]) == 3 and all(d["center"][1] == 560 for d in rows[1])


def test_split_rows_single_row_only():
    top = [make_det(x, 200) for x in (100, 500, 900)]
    rows = ga.split_rows_by_y(top, num_rows=2)
    assert len(rows[0]) == 3 and rows[1] == []
