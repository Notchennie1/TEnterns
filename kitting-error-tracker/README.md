# Kitting Error Tracker (Starter)

Base project structure for a computer-vision pipeline to track kitting errors using:
- MediaPipe hand tracking
- Segmentation for bin boundaries

## Project Structure

```text
kitting-error-tracker/
├─ configs/
│  └─ pipeline.yaml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
├─ scripts/
│  └─ run_local.py
├─ src/
│  └─ kitting_cv/
│     ├─ tracking/
│     │  └─ mediapipe_tracker.py
│     ├─ segmentation/
│     │  └─ bin_segmenter.py
│     └─ pipeline/
│        └─ run_pipeline.py
├─ requirements.txt
└─ pyproject.toml
```

## Quick Start (Windows PowerShell)

```powershell
cd kitting-error-tracker
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts/run_local.py
```

Press `q` to quit the camera window.

## What to implement next

1. Replace `BinSegmenter.segment(...)` in `src/kitting_cv/segmentation/bin_segmenter.py` with your model inference.
2. Add bin ID mapping (contour-to-bin labels).
3. Add logic to compare hand trajectory + picked item against expected kitting sequence.
4. Log errors (wrong bin, missed pick, extra pick) to a CSV or database.
