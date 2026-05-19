"""
YOLOv8 Segmentation Training Script — Bin Boundaries
=====================================================
Trains a YOLOv8-seg model to detect and segment bin boundaries
from annotated data (converted via coco_to_yolov8_seg.py).

Usage:
    python train_bin_segmentation.py --data path/to/data.yaml
    python train_bin_segmentation.py --data path/to/data.yaml --model yolov8n-seg.pt --epochs 150 --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train(
    data_yaml: str,
    model: str = "yolov8n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project: str = "runs/bin-seg",
    name: str = "train",
    resume: bool = False,
) -> Path:
    """
    Train YOLOv8 segmentation model.

    Returns the path to the best weights file.
    """
    from ultralytics import YOLO

    logger.info("Loading base model: %s", model)
    yolo = YOLO(model)

    logger.info("Starting training — %d epochs, imgsz=%d, batch=%d, device=%s",
                epochs, imgsz, batch, device)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        # Augmentation settings tuned for fixed-camera bin detection
        hsv_h=0.01,       # Minimal hue shift (bins don't change color much)
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=2.0,      # Slight rotation (camera might be slightly off)
        translate=0.05,    # Small translation
        scale=0.2,         # Some scale variation
        flipud=0.0,        # No vertical flip (bins have fixed orientation)
        fliplr=0.3,        # Occasional horizontal flip
        mosaic=0.5,        # Moderate mosaic augmentation
        patience=20,       # Early stopping patience
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    logger.info("Training complete! Best weights: %s", best_weights)
    return best_weights


def validate(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
) -> dict:
    """Run validation on a trained model."""
    from ultralytics import YOLO

    logger.info("Validating model: %s", model_path)
    yolo = YOLO(model_path)

    metrics = yolo.val(data=data_yaml, device=device, verbose=True)

    logger.info("Validation Results:")
    logger.info("  mAP50:    %.4f", metrics.seg.map50)
    logger.info("  mAP50-95: %.4f", metrics.seg.map)

    return {
        "map50": float(metrics.seg.map50),
        "map50_95": float(metrics.seg.map),
    }


def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    device: str = "cpu",
) -> Path:
    """Export trained model to deployment format."""
    from ultralytics import YOLO

    logger.info("Exporting model %s to %s format", model_path, format)
    yolo = YOLO(model_path)

    export_path = yolo.export(format=format, imgsz=imgsz, device=device)
    logger.info("Exported to: %s", export_path)
    return Path(export_path)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 bin segmentation model")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("--data", required=True, help="Path to data.yaml")
    train_p.add_argument("--model", default="yolov8n-seg.pt", help="Base model")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--imgsz", type=int, default=640)
    train_p.add_argument("--batch", type=int, default=16)
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument("--project", default="runs/bin-seg")
    train_p.add_argument("--name", default="train")
    train_p.add_argument("--resume", action="store_true")

    # Validate command
    val_p = sub.add_parser("val", help="Validate a trained model")
    val_p.add_argument("--model", required=True, help="Path to trained weights")
    val_p.add_argument("--data", required=True, help="Path to data.yaml")
    val_p.add_argument("--device", default="cpu")

    # Export command
    exp_p = sub.add_parser("export", help="Export model for deployment")
    exp_p.add_argument("--model", required=True, help="Path to trained weights")
    exp_p.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript"])
    exp_p.add_argument("--imgsz", type=int, default=640)
    exp_p.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        if not hasattr(args, "data") or args.data is None:
            parser.print_help()
            return
        train(
            data_yaml=args.data, model=args.model, epochs=args.epochs,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=args.project, name=args.name, resume=args.resume,
        )
    elif args.command == "val":
        validate(model_path=args.model, data_yaml=args.data, device=args.device)
    elif args.command == "export":
        export_model(model_path=args.model, format=args.format,
                     imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
