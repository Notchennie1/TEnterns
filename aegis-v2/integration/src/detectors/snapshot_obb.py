"""
Snapshot OBB driver
===================
Reads the aegis-core OBB handoff (``bins.json`` + ``snapshot.jpg``), allocates the
detections onto the fixed 1-9 grid via ``grid_allocator``, writes ``bins_indexed.json``,
and draws an overlay with the indices for visual verification.

Run (from aegis-v2/integration) as a PATH SCRIPT — this bypasses the detectors
package __init__, which eagerly imports cv2 / ultralytics:
    python src/detectors/snapshot_obb.py \
        --bins ../../aegis-core/runs/bins_obb_raw/bins.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Allow both ``python -m src.detectors.snapshot_obb`` and direct path import in tests.
sys.path.insert(0, os.path.dirname(__file__))
import grid_allocator as ga  # noqa: E402

logger = logging.getLogger("aegis.detectors.snapshot_obb")


def index_bins(payload: dict) -> dict:
    """Pure transform: raw handoff payload -> indexed grid dict.

    ``payload`` is the parsed ``bins.json`` ({"frame_w", "bins": [...]}). Returns
    {"bins": {bin_i: {...}}, "frame_w", "frame_h", "source", "rule"}.
    """
    frame_w = int(payload.get("frame_w") or 1280)
    detections = payload.get("bins", [])
    grid = ga.allocate_grid(detections, frame_w)
    return {
        "bins": grid,
        "frame_w": frame_w,
        "frame_h": int(payload.get("frame_h") or 0),
        "source": "obb",
        "rule": "rotate180_then_band_split",
    }


def _draw_overlay(image, grid: dict):
    import cv2
    import numpy as np
    vis = image.copy()
    for info in grid.values():
        idx = info["index"]
        if info["detected"]:
            pts = np.array(info["corners"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cx, cy = int(info["center"][0]), int(info["center"][1])
            cv2.putText(vis, str(idx), (cx - 10, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # (undetected cells are left unmarked on the snapshot)
    return vis


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    here = os.path.dirname(__file__)
    default_bins = os.path.abspath(os.path.join(
        here, "..", "..", "..", "..", "aegis-core", "runs", "bins_obb_raw", "bins.json"))

    ap = argparse.ArgumentParser(description="Allocate OBB snapshot bins to the 1-9 grid")
    ap.add_argument("--bins", default=default_bins, help="path to aegis-core bins.json")
    ap.add_argument("--out", default=None, help="output bins_indexed.json path")
    ap.add_argument("--no-show", action="store_true", help="don't open the overlay window")
    args = ap.parse_args()

    if not os.path.exists(args.bins):
        logger.error("bins.json not found: %s — run the aegis-core OBB script first.", args.bins)
        return
    try:
        with open(args.bins) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Could not read %s: %s", args.bins, e)
        return

    out = index_bins(payload)
    out_path = args.out or os.path.join(os.path.dirname(args.bins), "bins_indexed.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    filled = sum(1 for v in out["bins"].values() if v["detected"])
    logger.info("✓ Wrote %s (%d/9 cells filled)", out_path, filled)

    # Overlay on the sibling snapshot.jpg, if present.
    snap = os.path.join(os.path.dirname(args.bins), "snapshot.jpg")
    if os.path.exists(snap):
        import cv2
        image = cv2.imread(snap)
        if image is None:
            logger.error("Could not read snapshot image: %s — skipping overlay.", snap)
            return
        vis = _draw_overlay(image, out["bins"])
        vis_path = os.path.join(os.path.dirname(args.bins), "bins_indexed_overlay.jpg")
        cv2.imwrite(vis_path, vis)
        logger.info("✓ Wrote overlay %s", vis_path)
        if not args.no_show:
            cv2.imshow("Indexed bins (1-9) — 'q' to close", vis)
            while cv2.waitKey(20) & 0xFF != ord("q"):
                pass
            cv2.destroyAllWindows()
    else:
        logger.warning("No snapshot.jpg next to bins.json — skipping overlay.")


if __name__ == "__main__":
    main()
