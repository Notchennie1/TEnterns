"""Unit tests for grid_calibrator — pure geometry, no camera/model/cv2.

Imported by path so the test doesn't drag in the detectors package __init__
(which pulls cv2 / ultralytics).
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import pytest  # noqa: E402
import grid_calibrator as gc  # noqa: E402

FRAME_W = 1280


def make_det(cx, cy, conf=0.9, half=20):
    return {
        "corners": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
        "center": [float(cx), float(cy)],
        "conf": conf,
    }


def top_dets():
    # 6 bins across the top band (y=200), evenly spaced
    return [make_det(int((c + 0.5) * FRAME_W / 6), 200) for c in range(6)]


def bottom_dets():
    # 3 big bins across the bottom band (y=560)
    return [make_det(int((c + 0.5) * FRAME_W / 3), 560) for c in range(3)]


def full():
    return top_dets() + bottom_dets()


def test_calibrate_builds_9_indexed_slots():
    cal = gc.calibrate_grid(full())
    assert len(cal) == 9
    assert cal["slot_1"]["index"] == 1 and cal["slot_1"]["row"] == 0
    assert cal["slot_1"]["layer"] == "top" and cal["slot_1"]["span"] == 1
    assert cal["slot_6"]["index"] == 6 and cal["slot_6"]["row"] == 0
    assert cal["slot_7"]["index"] == 7 and cal["slot_7"]["row"] == 1
    assert cal["slot_7"]["layer"] == "bottom" and cal["slot_7"]["span"] == 2
    assert cal["slot_9"]["index"] == 9
    # each slot keeps its real centre + box + a positive per-row spacing
    assert cal["slot_1"]["center"][1] == 200 and cal["slot_7"]["center"][1] == 560
    assert "corners" in cal["slot_1"]
    assert cal["slot_1"]["row_spacing"] > 0 and cal["slot_7"]["row_spacing"] > 0


def test_calibrate_orders_left_to_right():
    cal = gc.calibrate_grid(list(reversed(top_dets())) + bottom_dets())
    xs = [cal[f"slot_{i}"]["center"][0] for i in range(1, 7)]
    assert xs == sorted(xs)  # slot_1..6 increase left->right regardless of input order


def test_calibrate_rejects_wrong_counts():
    with pytest.raises(ValueError):
        gc.calibrate_grid(top_dets()[:5] + bottom_dets())   # 5 top
    with pytest.raises(ValueError):
        gc.calibrate_grid(top_dets() + bottom_dets()[:2])   # 2 bottom


def test_calibrate_rejects_non_finite_center():
    bad = full()
    bad[0]["center"] = [float("nan"), 200.0]
    with pytest.raises(ValueError):
        gc.calibrate_grid(bad)


def test_calibrate_rejects_degenerate_spacing():
    # all 3 bottom bins detected at the same x -> zero row spacing -> reject
    dets = top_dets() + [make_det(640, 560) for _ in range(3)]
    with pytest.raises(ValueError):
        gc.calibrate_grid(dets)
