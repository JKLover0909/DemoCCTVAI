#!/usr/bin/env python3
"""
Phase 0: Inspect best.pt model to extract class names.
Outputs class list for DeepStream labels.txt generation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def inspect_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        print(f"ERROR: Model not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {path}")
    model = YOLO(str(path))

    names: dict = model.names  # {0: 'helmet', 1: 'vest', ...}

    print(f"\n{'='*50}")
    print(f"Model: {path.name}")
    print(f"Task:  {model.task}")
    try:
        nc = model.model.nc if hasattr(model.model, 'nc') else len(names)
        print(f"Classes (nc={nc}):")
    except Exception:
        print(f"Classes ({len(names)} total):")

    for idx, name in sorted(names.items()):
        print(f"  {idx:>3}: {name}")

    # Write labels.txt for DeepStream
    deepstream_labels_dir = PROJECT_ROOT / "deepstream"
    deepstream_labels_dir.mkdir(exist_ok=True)
    labels_path = deepstream_labels_dir / "labels.txt"

    with open(labels_path, "w") as f:
        for idx in sorted(names.keys()):
            f.write(f"{names[idx]}\n")

    print(f"\n✅ labels.txt written to: {labels_path}")
    print(f"   (one class per line, zero-indexed — required by DeepStream nvinfer)")
    print(f"\nClass list for reference:")
    print("  " + ", ".join(names[i] for i in sorted(names.keys())))
    print(f"{'='*50}\n")

    return names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect YOLO model classes")
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to model (relative to project root or absolute)",
    )
    args = parser.parse_args()
    inspect_model(args.model)
