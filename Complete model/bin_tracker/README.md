# Bin Tracker — Architecture Overview

A plug-and-play system that identifies which storage bin a human hand is reaching into. The system uses a fixed-position camera, a one-time bin boundary detection step, and continuous hand tracking to produce real-time bin assignment events displayed through a visual overlay.

This document describes the **software architecture only** — the structural design, data flow, module responsibilities, and extension points. It does not cover model training, dataset preparation, or weight files.

---

## Core Design Principle

The architecture is built around a single key insight: **the camera never moves**. Because the camera is fixed, bin boundaries only need to be detected once. After that initial snapshot, the boundary coordinates are locked for the entire session, and all subsequent processing is limited to tracking hand positions and mapping them against those fixed regions.

This two-phase design (detect once, track continuously) keeps the real-time loop lightweight — it never re-runs the expensive segmentation step.

---

## System Architecture

The system operates in three sequential stages, orchestrated by a central `Pipeline` class:

### Stage 1 — Initialization (runs once at startup)

When the system boots, it captures a single frame (the "snapshot") from the camera and passes it through the bin boundary detector. The detector produces a list of `BinRegion` objects, each containing a polygon (array of x,y vertices) that outlines one physical bin in pixel coordinates. These polygons are then converted into Shapely geometry objects and locked into the bin assignment engine. From this point forward, the bin map is immutable for the session.

The initialization stage touches three components in sequence:

1. **Camera** — opens the video source and captures the snapshot frame.
2. **BinBoundaryDetector** — loads a pre-trained segmentation model, runs inference on the snapshot, and extracts polygon boundaries from the segmentation masks (or falls back to rectangular bounding boxes if masks are unavailable).
3. **BinAssignmentEngine.set_bin_map()** — receives the detected `BinRegion` list and pre-builds Shapely `Polygon` objects for fast containment queries later.

### Stage 2 — Real-time Tracking Loop (runs continuously)

After initialization, the pipeline enters an infinite loop. Each iteration reads a frame from the camera, runs hand detection, and maps each detected hand to a bin:

1. **Camera Feed** — grabs the next frame from the video stream.
2. **Hand Tracker** — processes the frame and returns a list of `HandDetection` objects, each containing named landmarks (keypoints) with pixel coordinates.
3. **BinAssignmentEngine.assign()** — takes the hand detections, extracts the configured keypoint (by default `index_tip`), and tests whether that point falls inside any of the locked bin polygons. Returns a list of `BinEvent` objects.

The loop also logs FPS every 300 frames and supports a `q` keypress to quit.

### Stage 3 — UI Output (renders every frame)

The `OverlayUI` module takes the raw camera frame, the hand detections, and the bin events, and composites a visual overlay:

- **Bin boundary outlines** — each bin's polygon is drawn as a colored border. When a hand is inside a bin, that bin fills with a semi-transparent highlight.
- **Hand skeleton** — landmark keypoints are drawn as red dots connected by green lines following the anatomical hand structure (21 keypoints, 20 connections).
- **Status bar** — a dark bar at the top of the frame displays which bin each detected hand is currently reaching into (or "outside" if the hand isn't in any bin).

---

## Data Flow

The data flows through four clearly defined data structures that act as contracts between modules:

```
BinRegion          — output of detector, input to engine and UI
  ├── bin_id       (int)
  ├── label        (str, e.g. "bin_0")
  ├── polygon      (numpy array, shape Nx2)
  ├── centroid     (float tuple, auto-computed)
  └── confidence   (float)

HandDetection      — output of hand tracker, input to engine and UI
  ├── hand_id      (int)
  ├── handedness   ("left" | "right" | "unknown")
  ├── landmarks    (list of HandLandmark)
  │     ├── name   (str, e.g. "index_tip")
  │     ├── x, y   (float, pixel coordinates)
  │     ├── z      (float, depth if available)
  │     └── confidence (float)
  └── bounding_box (x1, y1, x2, y2 or None)

BinEvent           — output of engine, input to UI and logging
  ├── hand_id      (int)
  ├── handedness   (str)
  ├── bin_id       (int or None if outside all bins)
  ├── bin_label    (str or None)
  ├── hand_point   (float tuple, the tracked keypoint)
  ├── confidence   (float)
  └── method       ("point_in_polygon" | "nearest_centroid")
```

The pipeline flows like this each frame:

```
Camera Frame
    │
    ├──► BinBoundaryDetector  ──► list[BinRegion]  ──► (locked at startup)
    │         (Stage 1 only)
    │
    ├──► HandTracker.detect() ──► list[HandDetection]
    │                                    │
    │                                    ▼
    │                          BinAssignmentEngine.assign()
    │                                    │
    │                                    ▼
    │                             list[BinEvent]
    │                                    │
    └──► OverlayUI.render(frame, hands, events) ──► annotated frame ──► display
```

---

## Module Responsibilities

### `pipeline.py` — Pipeline Orchestrator

The central coordinator. It owns the lifecycle: loads config, opens the camera, runs initialization, enters the tracking loop, and handles cleanup. Every other module is instantiated and called from here. No module talks directly to another module — they all communicate through the pipeline via the shared data structures above.

Key methods:
- `run()` — the single entry point, runs the full lifecycle.
- `_initialize_bins()` — captures snapshot, runs detector, locks bin map.
- `_tracking_loop()` — the frame-by-frame processing loop.
- `_cleanup()` — releases camera, closes tracker, destroys windows.

### `detectors/bin_boundary_detector.py` — Bin Boundary Detector

Responsible for one thing: given a single image frame, produce a list of `BinRegion` polygons. It wraps the segmentation model and handles the conversion from raw model output (masks or boxes) into the standardized `BinRegion` format. After initialization, this module is never called again.

The detector supports two output modes depending on what the model provides: if segmentation masks are available, it extracts the mask contour as a polygon; if only bounding boxes are available, it constructs a rectangular polygon from the box coordinates.

### `trackers/base_hand_tracker.py` — Hand Tracker Interface

An abstract base class (`BaseHandTracker`) that defines the contract for any hand tracking backend. It requires two methods: `load_model()` to initialize the backend, and `detect(frame)` to return a list of `HandDetection` objects. There is also an optional `release()` hook for cleanup.

The `HandDetection` dataclass carries all the information about a detected hand: its landmarks (named keypoints with x,y coordinates), handedness, and bounding box. The `get_point(name)` convenience method lets downstream code extract a specific keypoint by name without iterating through the landmark list.

### `trackers/registry.py` — Tracker Registry

A class-level service locator that maps string names to tracker classes. When the pipeline reads `hand_tracker.backend: "mediapipe"` from config, it calls `TrackerRegistry.create("mediapipe", config)`, which looks up the class, instantiates it, calls `load_model()`, and returns the ready-to-use tracker.

New backends register themselves by calling `TrackerRegistry.register("name", MyClass)` — typically at module import time via a line at the bottom of their file.

### `trackers/mediapipe_tracker.py` — MediaPipe Backend (reference implementation)

A concrete implementation of `BaseHandTracker` that wraps Google's MediaPipe Hands. It converts MediaPipe's normalized landmark coordinates to pixel coordinates and maps the 21 MediaPipe keypoints to human-readable names (wrist, thumb_tip, index_tip, etc.). It auto-registers itself with the `TrackerRegistry` when its module is imported.

This file serves as the reference template for how to write a new backend.

### `engine/bin_assignment.py` — Bin Assignment Engine

The geometric decision-maker. It receives the locked bin map once (via `set_bin_map()`), pre-builds Shapely `Polygon` objects, and then on every frame tests whether each hand's tracked keypoint falls inside a bin polygon.

Two assignment strategies are available:
- **point_in_polygon** (default) — strict geometric containment. The hand must be inside the polygon boundary to register as "in" that bin. If the hand is outside all polygons, `bin_id` is `None`.
- **nearest_centroid** — a softer fallback that always assigns the hand to whichever bin's centroid is closest, even if the hand isn't geometrically inside any bin. Useful if bin boundaries are imprecise.

### `ui/overlay.py` — Overlay UI

Pure rendering logic. It takes a raw frame, a list of hand detections, and a list of bin events, and draws everything on a copy of the frame. It never modifies the source frame. Drawing is split into three layers: bin boundaries (with optional fill for active bins), hand skeletons, and a status bar. All rendering parameters (which layers to show, overlay transparency, etc.) come from the config.

### `utils/` — Config & Logging

- **config_loader.py** — reads the YAML config file, validates that all required sections are present, and returns a plain dictionary.
- **logger.py** — sets up a centralized logger with console output and optional file output, configured by the `logging` section of config.yaml.

---

## Plug-and-Play Extension Points

The architecture has two explicit extension points:

### 1. Hand Tracker Backends

To add a new hand tracking backend (e.g., YOLO-hand, a custom CNN, a depth-camera SDK):

1. Create a new file in `trackers/`, e.g. `trackers/yolo_hand_tracker.py`.
2. Subclass `BaseHandTracker`.
3. Implement `load_model()` (initialize your model/pipeline) and `detect(frame)` (return `list[HandDetection]`).
4. At the bottom of the file, call `TrackerRegistry.register("yolo_hand", YoloHandTracker)`.
5. In `config.yaml`, set `hand_tracker.backend: "yolo_hand"`.
6. Import the module in `pipeline.py` (one line) so it registers on startup.

No other code changes needed. The pipeline, engine, and UI are all backend-agnostic.

### 2. Assignment Strategies

The `BinAssignmentEngine` dispatches to a strategy method based on the `bin_assignment.method` config value. To add a new strategy:

1. Add a new private method `_assign_<name>()` in `bin_assignment.py`.
2. Add an `elif` branch in the `assign()` method to dispatch to it.
3. Set `bin_assignment.method: "<name>"` in config.

### 3. Configurable Keypoint

The engine reads `bin_assignment.hand_keypoint` from config to decide which landmark represents "the hand's reaching position." By default this is `index_tip`, but you can change it to `wrist`, `middle_tip`, or any other named landmark. No code change required — just update the config.

---

## File Structure

```
bin_tracker/
│
├── main.py                          # Entry point — parses args, runs Pipeline
├── pipeline.py                      # Orchestrator — wires all modules together
├── config.yaml                      # All runtime configuration
├── requirements.txt                 # Python dependencies
├── __init__.py
│
├── detectors/
│   ├── __init__.py
│   └── bin_boundary_detector.py     # YOLO-based one-shot bin detection
│
├── trackers/
│   ├── __init__.py
│   ├── base_hand_tracker.py         # Abstract interface + data structures
│   ├── registry.py                  # Service locator for backends
│   └── mediapipe_tracker.py         # Reference backend implementation
│
├── engine/
│   ├── __init__.py
│   └── bin_assignment.py            # Point-in-polygon / nearest-centroid logic
│
├── ui/
│   ├── __init__.py
│   └── overlay.py                   # OpenCV rendering of bins, hands, status
│
└── utils/
    ├── __init__.py
    ├── config_loader.py             # YAML config reader + validator
    └── logger.py                    # Centralized logging setup
```

---

## Configuration Reference

All runtime behavior is controlled by `config.yaml`. The key sections are:

- **camera** — video source (device index, file path, or RTSP URL), resolution, and FPS.
- **bin_detector** — path to segmentation weights, confidence/IOU thresholds, device selection, and whether to snapshot on startup.
- **hand_tracker** — which backend to use, confidence threshold, max number of hands to track.
- **bin_assignment** — which assignment strategy to use and which hand keypoint to track.
- **ui** — toggle for each overlay layer (bin boundaries, hand landmarks, active highlighting), transparency level, and window title.
- **logging** — log level and optional log file path.

---

## Architectural Decisions

**Why detect bins only once?** The camera is physically fixed, so bin positions in pixel space never change. Re-running segmentation every frame would waste compute for identical results. The one-shot approach turns an expensive per-frame cost into a negligible startup cost.

**Why abstract the hand tracker?** Hand tracking is the fastest-moving part of the CV stack. New models appear regularly, and different deployment environments may require different backends (MediaPipe for CPU, a custom model for GPU, a depth sensor SDK for 3D). The abstract interface means the rest of the system never needs to change when you swap backends.

**Why use Shapely for containment tests?** Shapely handles edge cases (self-intersecting polygons, points exactly on the boundary, degenerate shapes) that a naive ray-casting implementation would get wrong. The `buffer(0)` call auto-repairs invalid geometries from noisy segmentation output.

**Why a registry instead of a factory or config-driven import?** The registry pattern lets each backend register itself at import time with a single line. There's no central mapping to maintain, no import strings in config, and no risk of circular imports. Adding a backend is purely additive — you never touch existing files.

**Why dataclasses for data flow?** They provide typed, documented contracts between modules without the overhead of full ORM-style classes. Each module produces and consumes specific dataclasses, making the interfaces explicit and easy to test in isolation.
