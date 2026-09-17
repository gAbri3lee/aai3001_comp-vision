"""
Curates a hazard-detection dataset from Open Images V7 for the hazard
classes used throughout the pipeline (see src/decision_engine.py's
SEVERITY dict and src/live_pipeline_demo.py's HAZARD_CLASSES). This is the
"curated existing dataset" path documented in CLAUDE.md's Dataset rubric
section, not a raw unmodified dump.

Each class is pulled with its own capped query (only_matching=True) rather
than one combined query, so common classes like person/car don't crowd out
rarer ones like fire hydrant/stop sign in the sample budget. Results are
merged (fiftyone combines per-image label lists across queries) and
exported directly to YOLO format for ultralytics training.
"""
import argparse
import random
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

HAZARD_CLASS_MAP = {
    "person": "Person",
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "traffic light": "Traffic light",
    "fire hydrant": "Fire hydrant",
    "stop sign": "Stop sign",
    "bench": "Bench",
    "chair": "Chair",
    "dog": "Dog",
    "backpack": "Backpack",
    "suitcase": "Suitcase",
    "potted plant": "Houseplant",
    "couch": "Couch",
}

EXPORT_DIR = Path("data/hazard_dataset")


def build(per_class: int, val_fraction: float, seed: int):
    random.seed(seed)
    merged = fo.Dataset(name=f"hazard_merge_{random.randint(0, 1_000_000)}", persistent=False)

    for our_name, oi_name in HAZARD_CLASS_MAP.items():
        print(f"[build_hazard_dataset] pulling up to {per_class} images for '{oi_name}' ({our_name})...")
        # load_zoo_dataset caches/reuses a FiftyOne dataset by name, keyed on
        # (zoo dataset, split, max_samples) by default — NOT on `classes`.
        # Without an explicit unique dataset_name per class, every call here
        # after the first would silently reload the first class's cached
        # dataset instead of downloading a new one.
        slug = oi_name.lower().replace(" ", "_")
        subset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[oi_name],
            max_samples=per_class,
            only_matching=True,
            shuffle=True,
            seed=seed,
            dataset_name=f"oiv7_{slug}_{per_class}",
        )
        merged.merge_samples(subset, key_field="filepath")
        print(f"[build_hazard_dataset] merged dataset now has {len(merged)} unique images")
        fo.delete_dataset(subset.name)

    merged = merged.match(F("ground_truth.detections").length() > 0)
    print(f"[build_hazard_dataset] final image count: {len(merged)}")

    counts = merged.count_values("ground_truth.detections.label")
    print("[build_hazard_dataset] per-class box counts:")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:20} {count}")

    ids = merged.values("id")
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    classes = sorted(HAZARD_CLASS_MAP.values())
    train_view = merged.exclude(list(val_ids))
    val_view = merged.select(list(val_ids))

    train_view.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes,
        split="train",
    )
    val_view.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes,
        split="val",
    )

    print(f"[build_hazard_dataset] exported YOLO dataset to {EXPORT_DIR} "
          f"(train={len(train_view)}, val={len(val_view)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.per_class, args.val_fraction, args.seed)
