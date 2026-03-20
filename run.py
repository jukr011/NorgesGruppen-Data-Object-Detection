"""
NorgesGruppen Data: Object Detection
Competition submission entry point.

The server calls this script to run inference on shelf images.
It tries multiple input methods to be robust:
  1. --input <dir> --output <file>   (directory of images)
  2. --image <path>                  (single image)
  3. sys.argv[1:]                    (list of image paths)
  4. Reads IMAGE_DIR env var as fallback
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


# ── helpers ───────────────────────────────────────────────────────────────────

def find_model_weights() -> str:
    """Return path to the best available YOLO weights file."""
    candidates = [
        "best.pt",
        "weights/best.pt",
        "model/best.pt",
        "last.pt",
        "weights/last.pt",
        "yolov8s.pt",   # pretrained baseline fallback
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No model weights found. Expected best.pt or yolov8s.pt in the zip root."
    )


def collect_images(paths: list[str]) -> list[Path]:
    """Expand directories and filter image files."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for ext in IMG_EXTS:
                images.extend(sorted(p.glob(f"*{ext}")))
                images.extend(sorted(p.glob(f"*{ext.upper()}")))
        elif p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return images


def image_id_from_path(img_path: Path) -> int | str:
    """Try to parse a numeric image id from the filename, else return the stem."""
    stem = img_path.stem
    # e.g. img_00042 → 42
    digits = "".join(filter(str.isdigit, stem))
    return int(digits) if digits else stem


# ── main inference ─────────────────────────────────────────────────────────────

def run_inference(image_paths: list[Path], model: YOLO, conf: float = 0.25) -> list[dict]:
    """Run YOLO inference and return COCO-format result list."""
    results = []

    for img_path in image_paths:
        img_id = image_id_from_path(img_path)
        preds = model.predict(
            source=str(img_path),
            conf=conf,
            iou=0.45,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        for pred in preds:
            boxes = pred.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(box.cls[0].item()),
                        "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        "score": round(float(box.conf[0].item()), 4),
                    }
                )

    return results


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NorgesGruppen object detection inference")
    parser.add_argument("--input", type=str, help="Directory with input images")
    parser.add_argument("--output", type=str, help="Path to write predictions JSON")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args, extra = parser.parse_known_args()

    # Determine image sources
    sources: list[str] = []
    if args.input:
        sources.append(args.input)
    if args.image:
        sources.append(args.image)
    if extra:
        sources.extend(extra)
    if not sources:
        env_dir = os.environ.get("IMAGE_DIR", "")
        if env_dir:
            sources.append(env_dir)
    if not sources:
        print("ERROR: No image source provided. Use --input <dir> or --image <path>", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_images(sources)
    if not image_paths:
        print(f"WARNING: No images found in {sources}", file=sys.stderr)
        predictions = []
    else:
        print(f"Running inference on {len(image_paths)} image(s)...", file=sys.stderr)
        weights = find_model_weights()
        print(f"Loading model from {weights}", file=sys.stderr)
        model = YOLO(weights)
        predictions = run_inference(image_paths, model, conf=args.conf)
        print(f"Generated {len(predictions)} detections.", file=sys.stderr)

    # Output
    output_json = json.dumps(predictions, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Predictions written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
