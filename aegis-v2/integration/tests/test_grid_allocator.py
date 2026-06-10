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
