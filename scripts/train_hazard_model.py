"""
Fine-tunes YOLOv8n on the curated hazard dataset (see build_hazard_dataset.py)
starting from COCO-pretrained weights. This is the actual fine-tuning step
for the Model Development rubric row.

CPU-only, so epochs/imgsz are kept modest here for a baseline checkpoint —
increase both once more time/hardware is available, before treating this as
a final submission model (see CLAUDE.md's model swap strategy: this output
is what gets passed as --model to the live pipeline / webapp).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/hazard_dataset/dataset.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--name", default="hazard_yolov8n")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} not found - run scripts/build_hazard_dataset.py first")

    model = YOLO(args.base_model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        name=args.name,
        patience=10,
    )
    print(f"Best checkpoint: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
