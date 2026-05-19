# AEGIS v2 — Bin Tracking & Kinetic Gating System

Real-time computer vision system that detects bin boundaries, tracks hand position, and determines which bin a hand is reaching into — enabling kinetic gating for load receptor activation in manual kitting workflows.

## How It Works

The system runs in three stages:

**1. Bin Detection (startup)** — A YOLOv8 segmentation model identifies bin boundaries from a single camera snapshot. The detected polygons/bounding boxes are locked as the bin map for the session.

**2. Hand Tracking (real-time)** — A hand recognition model (MediaPipe by default, swappable) detects hand position, landmarks, and grab gestures every frame.

**3. Integration (real-time)** — The bin assignment engine checks whether the hand's position falls inside a bin region. The Triple-Gate FSM then runs kinetic gating: Spatial (hand in bin) → Intent (grab detected) → Verification (weight change) → load receptor activation.

```
Camera Frame
    │
    ├──► Bin Detector (YOLOv8-seg)  ──► Bin Map (x,y polygons)
    │         [runs once at startup]          │
    │                                         ▼
    └──► Hand Tracker (MediaPipe/YOLO) ──► Bin Assignment Engine
              [runs every frame]              │
                                              ▼
                                     Triple-Gate FSM
                                              │
                                              ▼
                                   Load Receptor Activation
```

## Project Structure

```
aegis-v2/
│
├── cv-models/                    # Bin boundary detection (YOLOv8 segmentation)
│   ├── scripts/
│   │   ├── coco_to_yolov8_seg.py       # Convert Roboflow COCO annotations → YOLOv8-seg format
│   │   ├── train_bin_segmentation.py   # Train / validate / export the bin detector
│   │   └── inference_bin_detector.py   # Standalone inference & visualization
│   ├── configs/
│   │   └── training_config.yaml        # Training hyperparameters
│   ├── data/
│   │   ├── raw/                        # Raw images from Roboflow
│   │   ├── annotations/               # COCO JSON annotation files
│   │   └── processed/                 # Converted YOLOv8-seg dataset (auto-generated)
│   ├── models/weights/                 # Trained .pt weights go here
│   └── requirements.txt
│
├── hand-models/                  # Hand recognition experiments (multiple backends)
│   ├── common/
│   │   ├── base_hand_tracker.py        # Abstract interface all backends implement
│   │   └── registry.py                 # Service locator — swap backends by name
│   ├── mediapipe/
│   │   └── tracker.py                  # MediaPipe Tasks Hand Landmarker (primary)
│   ├── yolo-hand/
│   │   ├── tracker.py                  # YOLOv8-pose hand tracker (experimental)
│   │   └── weights/                    # Place YOLO hand model weights here
│   ├── custom-model-template/
│   │   └── tracker.py                  # Copy this folder to add a new backend
│   ├── benchmarks/
│   │   └── run_benchmark.py            # Side-by-side latency/accuracy comparison
│   └── requirements.txt
│
├── integration/                  # Combined pipeline (wires cv-models + hand-models + UI)
│   ├── src/
│   │   ├── pipeline.py                 # Main orchestrator (Sense → Analyse → Act)
│   │   ├── detectors/
│   │   │   └── bin_detector.py         # Adapter wrapping cv-models for the pipeline
│   │   ├── engine/
│   │   │   ├── bin_assignment.py       # Assigns hands to bins (point-in-polygon / overlap)
│   │   │   └── fsm.py                 # Triple-Gate FSM for kinetic gating
│   │   └── ui/
│   │       ├── overlay.py              # OpenCV real-time camera overlay
│   │       ├── dashboard.py            # FastAPI web dashboard backend
│   │       ├── state.py                # Thread-safe shared state (pipeline ↔ dashboard)
│   │       └── static/                 # Dashboard frontend (HTML/CSS/JS)
│   │           ├── index.html
│   │           ├── style.css
│   │           └── app.js
│   ├── config/
│   │   └── settings.yaml               # Master config (camera, models, FSM, UI, sensing)
│   ├── tests/
│   └── requirements.txt
│
└── README.md                     # This file
```

## Quick Start

### 1. Set Up Environment

```bash
cd aegis-v2
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r integration/requirements.txt
```

### 2. Prepare the Bin Detection Model

**If you already have a trained model**, place `best.pt` in `cv-models/models/weights/`.

**To train from scratch using Roboflow annotations:**

```bash
# Step 1: Export your dataset from Roboflow in COCO Segmentation format
# Place the images in cv-models/data/raw/
# Place _annotations.coco.json in cv-models/data/annotations/

# Step 2: Convert COCO → YOLOv8-seg format
python cv-models/scripts/coco_to_yolov8_seg.py \
    --coco-json cv-models/data/annotations/_annotations.coco.json \
    --images-dir cv-models/data/raw/ \
    --output-dir cv-models/data/processed/

# Step 3: Train
python cv-models/scripts/train_bin_segmentation.py train \
    --data cv-models/data/processed/data.yaml \
    --model yolov8n-seg.pt \
    --epochs 100 \
    --device cuda:0

# Step 4: Copy best weights
cp runs/bin-seg/train/weights/best.pt cv-models/models/weights/
```

### 3. Set Up Hand Tracking

The MediaPipe backend works out of the box. Download the model file:

```bash
# Download hand_landmarker.task (float16, ~10MB)
curl -o hand-models/mediapipe/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 4. Run the Pipeline

```bash
python -m integration.src.pipeline --config integration/config/settings.yaml
```

Press `q` to quit the camera overlay. The web dashboard launches automatically at `http://localhost:8080`.

### 5. (Optional) Preview the Dashboard Without a Camera

For UI-only work — tweaking layout, styling, error messaging — you can run the dashboard with synthetic data. No camera, no YOLO model, no MediaPipe required:

```powershell
cd aegis-v2
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" opencv-python-headless numpy
py scripts\dev_mock.py
```

Then open `http://localhost:8080`. The mock feeds fake bins, hands, FSM transitions, picks, and errors into `PipelineState` so every section of the dashboard populates. Useful for previewing the operator UI without the full CV stack.

## Operator UI

AEGIS v2 has two UI layers that run simultaneously:

**OpenCV Camera Overlay** — drawn directly on the live camera feed, showing bin boundaries, hand skeletons, grab gesture indicators, and a real-time FSM gate progress badge. This is the low-latency view for debugging and development.

**Web Dashboard** — a FastAPI-served browser dashboard at `http://localhost:8080` for the operator. It shows a color-coded bin grid (white → orange → green as picks progress), hand tracking cards with grab status, FSM gate visualization, system stats (FPS, uptime, frame count), and full-screen error alerts when a gate fails.

Both are enabled by default. To disable either, edit `integration/config/settings.yaml`:

```yaml
ui:
  enabled: true       # OpenCV overlay (set false for headless)

dashboard:
  enabled: true       # Web dashboard
  port: 8080
```

The dashboard reads from a thread-safe shared state object that the pipeline loop writes to every frame, so there's no performance coupling between the web server and the CV loop.

## Plugging In Your Own Models

The repo is designed around two clean integration points: a bin (CV) model and a hand model. Each connects to the runtime in a different way.

### Plugging In Your Own Bin Model

The bin detector is selected by config path — no code changes needed if your model is a YOLOv8-seg `.pt` file.

1. Train in `cv-models/` (or bring your own weights).
2. Drop the weights in `cv-models/models/weights/best.pt` (or anywhere — just remember the path).
3. Update `integration/config/settings.yaml`:
   ```yaml
   bin_detector:
     model_path: "../cv-models/models/weights/best.pt"
     confidence_threshold: 0.5
   ```
4. Run the pipeline.

**Gotcha — bin grid layout:** `integration/src/detectors/bin_detector.py` currently assumes a **4-column grid** and assigns IDs via `bin_{row}_{col}` where `row, col = rank // 4, rank % 4` (see `_sort_spatially`). If your bin layout is different (3×3, 2×6, irregular), edit that block. The IDs flow through to the dashboard and FSM events, so renaming there is enough.

### Switching Hand Models

If a backend is already registered, just edit `integration/config/settings.yaml`:

```yaml
hand_tracker:
  backend: "mediapipe"    # or "yolo-hand", or your custom backend name
```

All registered backends are listed at startup.

### Adding a New Hand Model

Hand models use a **registry pattern** — they don't just drop into a folder. You implement an interface, register the class, and flip a config value.

1. **Copy the template:**
   ```powershell
   Copy-Item -Recurse hand-models\custom-model-template hand-models\my-model
   ```
2. **Edit `hand-models/my-model/tracker.py`:**
   - Rename `CustomHandTracker` → `MyTracker`
   - Implement `load_model()` — load your weights (ONNX, PyTorch, TF, whatever)
   - Implement `detect(frame: np.ndarray) -> list[HandDetection]` — return `HandDetection` objects with at minimum `hand_id`, `handedness`, `landmarks` (must include `wrist` and `index_tip`), and `bounding_box`. Optionally set `is_grabbing` and `grab_score`.
   - Uncomment the bottom line: `TrackerRegistry.register("my-model", MyTracker)`
3. **Register the import in `integration/src/pipeline.py`** (in the auto-register block near the top):
   ```python
   try:
       import hand_models.my_model.tracker  # noqa: F401
   except ImportError:
       pass
   ```
   Note: hyphens in folder names become underscores in the Python import path (`my-model` → `my_model`).
4. **Activate it in config:**
   ```yaml
   hand_tracker:
     backend: "my-model"
   ```

The contract you must satisfy lives in `hand-models/common/base_hand_tracker.py` (the `HandDetection` / `HandLandmark` dataclasses). As long as your tracker returns those shapes, the bin assignment engine, FSM, overlay, and dashboard all work without modification.

## Benchmarking Hand Models

Compare all registered backends side-by-side:

```bash
python hand-models/benchmarks/run_benchmark.py --source 0 --frames 300
```

Output includes latency (ms), FPS, detection rate, and landmark jitter for each backend.

## Triple-Gate FSM (Kinetic Gating)

The FSM controls when load receptors are activated:

| Gate | Trigger | Sensor | Timeout |
|------|---------|--------|---------|
| 1. Spatial | Hand enters bin geofence | CV bin assignment | 2.0s |
| 2. Intent | Grab gesture detected | Hand model (closed fist) | 1.5s |
| 3. Verify | Weight change exceeds threshold | Modbus load cell (future) | 3.0s |

All three gates must pass in sequence. On success, the load receptor for that bin is activated. On timeout, the system resets with an error cooldown.

## What Changed from aegis-core

`aegis-v2` is a reorganized successor to `aegis-core`. The original `aegis-core` folder is preserved untouched so you can diff against it. The high-level idea: v1 was one monolithic `src/` tree; v2 splits it into three independent sibling packages (`cv-models/`, `hand-models/`, `integration/`), each with its own `requirements.txt`.

### Structural changes

| aegis-core (v1)                              | aegis-v2                                            | Notes |
|----------------------------------------------|------------------------------------------------------|-------|
| `src/vision/bin_detector.py`                 | `cv-models/` (training) + `integration/src/detectors/bin_detector.py` (runtime adapter) | Training and runtime split into separate packages |
| `src/vision/geofencing.py` (`DynamicGeofenceManager`) | Folded into `integration/src/detectors/bin_detector.py` | Same class, same behavior |
| `src/trackers/{base,registry,mediapipe}.py`  | `hand-models/common/` + `hand-models/mediapipe/`     | Each backend now lives in its own folder |
| `src/main.py` (CLI entry)                    | `integration/src/pipeline.py` has its own `__main__` | One less file |
| `src/utils/{config_loader,logger}.py`        | Inlined into `pipeline.py` (`_load_config`, `_setup_logging`) | Trivial helpers — didn't need their own modules |
| `src/ui/backend.py`                          | `integration/src/ui/dashboard.py`                    | FastAPI app, richer endpoints |
| `src/logic/fsm.py` (older duplicate)         | Removed; canonical FSM is `integration/src/engine/fsm.py` | Only one FSM now |

### Behavioral / API changes

- **Bin assignment no longer requires Shapely.** `engine/bin_assignment.py` now uses inline bounding-box checks; added a third method `area_overlap` (config: `bin_assignment.method`).
- **FSM gained `set_callbacks(on_success, on_error)`.** The pipeline subscribes to these to record picks and push errors into the dashboard.
- **Bin detector visualize method renamed**: `visualize_detections` → `visualize`.
- **Default detection confidence**: 0.6 → 0.5.
- **Bin grid assumption**: v1 had spatially-sorted row/column logic; v2's `_sort_spatially` keeps row-major sort but **hardcodes 4 columns** in `bin_{row}_{col}` naming (see Plugging In Your Own Models above).

### New in v2

- `integration/src/ui/dashboard.py` — FastAPI dashboard backend (replaces and expands `src/ui/backend.py`).
- `integration/src/ui/state.py` — thread-safe `PipelineState` carrying live bins/hands/FSM/errors/stats. The bridge between the CV loop and the web UI.
- `integration/src/ui/static/` — ~3× the size of v1's frontend. Adds FSM gate banner, hand cards, system stats row, FPS chip, and dismissable error overlay.
- `cv-models/scripts/coco_to_yolov8_seg.py` — Roboflow COCO-segmentation → YOLOv8-seg dataset converter.
- `hand-models/benchmarks/run_benchmark.py` — side-by-side latency/accuracy comparison.
- `hand-models/custom-model-template/` — copy-paste starting point for new backends.
- `scripts/dev_mock.py` — fake-data driver for previewing the dashboard without a camera (see Quick Start §5).

### Known gaps (worth flagging to peers)

- **Load-cell weight sensing is not wired.** `aegis-core/src/sensing/modbus_client.py` (189-line async pymodbus client) has no v2 equivalent. The FSM Gate 3 currently receives `weight_delta=0.0` hardcoded in `pipeline.py` (search for the `# TODO: Modbus weight sensor` line). Gate 3 will only pass when its timeout-based fallback fires — full weight verification needs to be re-implemented.
- **Empty stub packages.** `integration/src/{logic, sensing, trackers, utils}/` exist with only a one-line `__init__.py`. They're placeholders, not broken — the actual logic moved into `engine/`, `hand-models/`, and `pipeline.py`.
- **Geofence smoothing is available but unused at runtime.** `DynamicGeofenceManager` is defined in `bin_detector.py` but the pipeline runs a single startup snapshot only.

## Hardware Requirements

- **Camera**: Any USB webcam (1280x720 recommended)
- **GPU**: Optional — CPU works for development; CUDA GPU recommended for training and deployment
- **Sensors**: Modbus TCP load cells (future integration, not yet wired)

## Target Performance

- End-to-end latency: <500ms
- Frame rate: 30 FPS
- Deployment: NVIDIA Jetson / edge AI hardware with INT8 quantization
