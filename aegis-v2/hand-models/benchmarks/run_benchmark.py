"""
Hand Model Benchmark Runner
============================
Runs side-by-side performance comparison of all registered hand trackers.

Measures:
  - Inference latency (ms per frame)
  - Detection rate (% frames with hands detected)
  - Landmark stability (jitter across frames)
  - FPS throughput

Usage:
    # Benchmark all registered backends on webcam
    python run_benchmark.py --source 0 --frames 300

    # Benchmark specific backend on a video file
    python run_benchmark.py --backend mediapipe --source test_video.mp4

    # Compare two backends
    python run_benchmark.py --backend mediapipe yolo-hand --source 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to path so we can import hand_models
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hand_models.common import TrackerRegistry, HandDetection

# Import all backends to trigger auto-registration
try:
    import hand_models.mediapipe.tracker  # noqa: F401
except ImportError:
    pass
try:
    import hand_models.yolo_hand.tracker  # noqa: F401
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def benchmark_backend(
    backend_name: str,
    config: dict,
    source: int | str,
    num_frames: int,
) -> dict:
    """
    Run benchmark for a single backend.

    Returns a dict with latency, detection rate, and FPS stats.
    """
    logger.info("=" * 50)
    logger.info("Benchmarking: %s", backend_name)
    logger.info("=" * 50)

    try:
        tracker = TrackerRegistry.create(backend_name, config)
    except Exception as e:
        logger.error("Failed to create tracker '%s': %s", backend_name, e)
        return {"backend": backend_name, "error": str(e)}

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        return {"backend": backend_name, "error": f"Cannot open source: {source}"}

    latencies: list[float] = []
    detections_per_frame: list[int] = []
    landmark_positions: list[list[tuple[float, float]]] = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        hands = tracker.detect(frame)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms
        detections_per_frame.append(len(hands))

        # Track landmark positions for stability analysis
        if hands:
            pts = []
            for h in hands:
                tip = h.get_point("index_tip")
                if tip:
                    pts.append(tip)
            landmark_positions.append(pts)

        if (i + 1) % 50 == 0:
            logger.info("  Frame %d/%d — avg latency: %.1f ms",
                        i + 1, num_frames, np.mean(latencies[-50:]))

    cap.release()
    tracker.release()

    # Compute stats
    latencies_arr = np.array(latencies)
    detection_rate = sum(1 for d in detections_per_frame if d > 0) / len(detections_per_frame) * 100

    # Landmark jitter (std of frame-to-frame movement)
    jitter = 0.0
    if len(landmark_positions) > 1:
        deltas = []
        for i in range(1, len(landmark_positions)):
            if landmark_positions[i] and landmark_positions[i - 1]:
                for curr, prev in zip(landmark_positions[i], landmark_positions[i - 1]):
                    dx = curr[0] - prev[0]
                    dy = curr[1] - prev[1]
                    deltas.append(np.sqrt(dx**2 + dy**2))
        if deltas:
            jitter = float(np.std(deltas))

    results = {
        "backend": backend_name,
        "frames": len(latencies),
        "latency_mean_ms": float(np.mean(latencies_arr)),
        "latency_median_ms": float(np.median(latencies_arr)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_max_ms": float(np.max(latencies_arr)),
        "fps": 1000.0 / float(np.mean(latencies_arr)),
        "detection_rate_pct": detection_rate,
        "landmark_jitter_px": jitter,
    }

    logger.info("Results for %s:", backend_name)
    for k, v in results.items():
        if k != "backend":
            logger.info("  %s: %s", k, f"{v:.2f}" if isinstance(v, float) else v)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark hand tracking models")
    parser.add_argument("--backend", nargs="*", default=None,
                        help="Backend name(s) to test. Default: all registered.")
    parser.add_argument("--source", default="0", help="Video source (camera index or file path)")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to process")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=2)
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    config = {
        "confidence_threshold": args.confidence,
        "max_hands": args.max_hands,
    }

    backends = args.backend or TrackerRegistry.available()
    if not backends:
        logger.error("No backends registered. Import tracker modules first.")
        return

    all_results = []
    for name in backends:
        result = benchmark_backend(name, config, source, args.frames)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Backend':<20} {'Latency(ms)':<14} {'FPS':<8} {'Detect%':<10} {'Jitter(px)':<10}")
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"{r['backend']:<20} ERROR: {r['error']}")
        else:
            print(f"{r['backend']:<20} {r['latency_mean_ms']:<14.1f} "
                  f"{r['fps']:<8.1f} {r['detection_rate_pct']:<10.1f} "
                  f"{r['landmark_jitter_px']:<10.1f}")

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
