import subprocess
import sys

# Ensure dependencies are available in evaluation environment
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Python: {sys.version}", flush=True)
    print(f"Input: {args.input}", flush=True)
    print(f"Output: {args.output}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    model_path = Path(__file__).parent / "best.pt"
    print(f"Model: {model_path} (exists={model_path.exists()})", flush=True)
    model = YOLO(str(model_path))

    input_path = Path(args.input)
    images = sorted([f for f in input_path.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"Found {len(images)} images", flush=True)

    predictions = []
    for img in images:
        try:
            stem = img.stem
            parts = stem.split("_")
            image_id = int(parts[-1])
        except (ValueError, IndexError):
            try:
                image_id = int(stem)
            except ValueError:
                print(f"WARNING: cannot parse image_id from '{img.name}', skipping", flush=True)
                continue

        try:
            results = model(str(img), device=device, verbose=False, conf=args.conf)
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(r.boxes.cls[i].item()),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                        "score": round(float(r.boxes.conf[i].item()), 3),
                    })
        except Exception as e:
            print(f"ERROR on {img.name}: {e}", flush=True)
            continue

    print(f"Total detections: {len(predictions)}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
