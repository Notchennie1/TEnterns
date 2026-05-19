# kitting-error-tracker

Minimal local pipeline for webcam-based hand landmark tracking (Google AI Edge / MediaPipe Tasks) + a simple bin mask preview.

## What this project does

- Opens a webcam feed.
- Runs **MediaPipe Tasks Hand Landmarker** (from the `mediapipe` Python package) to detect hand landmarks.
- Computes a simple **GRAB / OPEN** state using a normalized *finger curl* heuristic + temporal debouncing.
- Displays:
  - `kitting-camera`: the camera frame with landmarks and text overlay
  - `bin-mask`: a basic segmentation mask (placeholder)

### GRAB / OPEN meaning

`GRAB` is estimated with a simple closed-fist proxy:

- Compute the palm center from wrist + MCP joints.
- Measure each fingertip’s distance to the palm center.
- Normalize those distances by palm size (wrist ↔ middle-MCP).
- If **4/5** fingertips are “close” (curled), report `GRAB`.

This is then temporally debounced across frames to reduce flicker.

## Requirements

- Windows (these instructions are written for Windows PowerShell)
- Conda (recommended) or Python 3.10 + venv
- A webcam

## First-time setup (Conda)

From the repository root, run:

```powershell
cd kitting-error-tracker
conda env create -f environment.yml
conda activate kitting-cv
```

### Download the Hand Landmarker model

The tracker expects this file:

- `kitting-error-tracker/models/hand_landmarker.task`

Download it with:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models/hand_landmarker.task"
```

Quick check:

```powershell
Test-Path models/hand_landmarker.task
```

### Smoke test imports

```powershell
python -c "import cv2, mediapipe as mp; print('imports ok', cv2.__version__, mp.__version__)"
```

## Run (first time and later)

Activate the environment and run:

```powershell
cd kitting-error-tracker
conda activate kitting-cv
python scripts/run_local.py
```

Notes:
- Press `q` in the OpenCV window to quit.
- If you accidentally run `python scripts/run_local.pyd`, that will fail: the script is `run_local.py`.

## Run again (testing the same model)

For subsequent runs on the same model, you only need:

```powershell
cd kitting-error-tracker
conda activate kitting-cv
python scripts/run_local.py
```

(Optional) confirm the model is still present:

```powershell
Test-Path models/hand_landmarker.task
```

## Configuration

### Camera index

The entrypoint is:

- `scripts/run_local.py`

It calls `run_camera_pipeline(camera_index=0)`. If your webcam is not at index 0, edit that value (e.g. `1`).

## Troubleshooting

### "Import 'cv2' could not be resolved" / "Import 'mediapipe' could not be resolved" in VS Code

VS Code is using a different Python interpreter than your conda env.

Fix:
- Open the Command Palette → **Python: Select Interpreter**
- Pick the interpreter from the `kitting-cv` environment

### Model not found

If you see an error like `FileNotFoundError: Hand Landmarker model not found...`, ensure:

- `models/hand_landmarker.task` exists
- You ran the download command in the `kitting-error-tracker/` folder

### Camera won’t open

If you get `Could not open camera.`:

- Close other apps that may be using the camera.
- Try a different `camera_index` in `scripts/run_local.py`.

## Project layout

- `src/kitting_cv/tracking/mediapipe_tracker.py`: Tasks Hand Landmarker + GRAB/OPEN logic
- `src/kitting_cv/pipeline/run_pipeline.py`: webcam loop + drawing/visualization
- `src/kitting_cv/segmentation/bin_segmenter.py`: placeholder segmentation
- `scripts/run_local.py`: entrypoint
