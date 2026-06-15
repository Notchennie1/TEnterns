"""Unit tests for the occlusion gate inside BinAssignmentEngine.

Pure geometry — no camera/model/cv2. HandDetection/HandLandmark are imported
straight from base_hand_tracker (numpy-only) so the detectors package __init__
(which pulls cv2 / ultralytics) is never touched.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import pytest  # noqa: E402
from integration.src.engine.bin_assignment import (  # noqa: E402
    BinAssignmentEngine, BinRegion,
)
from hand_models.common.base_hand_tracker import (  # noqa: E402
    HandDetection, HandLandmark,
)


# Bin map: top row (row 0) of 4 single bins across x, bottom row (row 1) of
# 2 wide bins. Top bins occupy y 0..100; bottom bins y 200..400 (rim at y=200).
def make_bins():
    top = [
        BinRegion(bin_id=f"bin_0_{c}", label=f"bin_0_{c}",
                  x_min=c * 100, x_max=c * 100 + 100,
                  y_min=0, y_max=100, confidence=0.9)
        for c in range(4)
    ]
    bottom = [
        BinRegion(bin_id="bin_1_0", label="bin_1_0",
                  x_min=0, x_max=200, y_min=200, y_max=400, confidence=0.8),
        BinRegion(bin_id="bin_1_1", label="bin_1_1",
                  x_min=200, x_max=400, y_min=200, y_max=400, confidence=0.7),
    ]
    return top + bottom


def make_engine(enabled=True):
    return BinAssignmentEngine({
        "method": "point_in_polygon",
        "hand_keypoint": "index_tip",
        "occlusion_gate": {"enabled": enabled},
    })


def test_grid_structure_precomputed_on_set_bin_map():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    assert eng._top_rows == {0}
    assert [b.bin_id for b in eng._bottom_bins] == ["bin_1_0", "bin_1_1"]
    assert eng._global_occ_y == 200


def test_single_row_layout_makes_gate_inert():
    eng = make_engine()
    eng.set_bin_map([
        BinRegion(bin_id="bin_0_0", label="bin_0_0",
                  x_min=0, x_max=100, y_min=0, y_max=100),
    ])
    assert eng._top_rows == set()
    assert eng._bottom_bins == []
