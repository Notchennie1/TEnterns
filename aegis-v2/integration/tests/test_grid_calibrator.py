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


def present_indices(occ):
    return {info["index"] for info in occ.values() if info["present"]}


def test_match_all_present():
    cal = gc.calibrate_grid(full())
    occ = gc.match_to_grid(full(), cal)
    assert present_indices(occ) == set(range(1, 10))
    assert occ["slot_1"]["present"] is True
    assert occ["slot_1"]["confidence"] == 0.9
    assert "corners" in occ["slot_1"]


def test_match_missing_middle_top_bin():
    cal = gc.calibrate_grid(full())
    dets = [d for c, d in enumerate(top_dets()) if c != 2] + bottom_dets()  # drop top index 3
    occ = gc.match_to_grid(dets, cal)
    assert occ["slot_3"]["present"] is False
    assert occ["slot_2"]["present"] is True and occ["slot_4"]["present"] is True
    assert present_indices(occ) == {1, 2, 4, 5, 6, 7, 8, 9}


def test_match_tolerates_small_jitter():
    cal = gc.calibrate_grid(full())
    jittered = [make_det(d["center"][0] + 8, d["center"][1] - 6) for d in full()]
    occ = gc.match_to_grid(jittered, cal)
    assert present_indices(occ) == set(range(1, 10))


def test_match_drops_stray_far_detection():
    cal = gc.calibrate_grid(full())
    stray = make_det(30, 30)  # top-left corner, far from every slot
    occ = gc.match_to_grid(full() + [stray], cal)
    assert present_indices(occ) == set(range(1, 10))  # 9 reals matched, stray dropped


def test_match_uses_nearest_center_not_index_order():
    cal = gc.calibrate_grid(full())
    # one bin sitting on slot_2's calibrated centre -> must match slot_2, not slot_1
    s2 = cal["slot_2"]["center"]
    occ = gc.match_to_grid([make_det(s2[0], s2[1])], cal)
    assert occ["slot_2"]["present"] is True
    assert occ["slot_1"]["present"] is False
