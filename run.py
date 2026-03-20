"""
NorgesGruppen Data: Object Detection
Competition submission entry point.

The server calls:
    python run.py --input /data/images --output /output/predictions.json

Features:
- Tiled inference: slices 2000x1500 shelf images into 640px tiles so small
  products are detected at full resolution, then merges with batched NMS.
- Reference re-ranking: if reference_embeddings.npy + reference_labels.json +
  feature_extractor.pt are present, uses MobileNetV3 similarity to correct
  category assignments after detection.

Sandbox-safe: no import os / sys / yaml / pickle / sahi / shutil.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import batched_nms
from ultralytics import YOLO


# ── weight file discovery ─────────────────────────────────────────────────────

def find_model_weights() -> str:
    candidates = ["best.pt", "weights/best.pt", "model/best.pt", "last.pt", "yolov8s.pt"]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "No model weights found. Expected best.pt or yolov8s.pt in the zip root."
    )


# ── image helpers ─────────────────────────────────────────────────────────────

def collect_images(paths: list[str]) -> list[Path]:
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


def image_id_from_path(img_path: Path) -> int:
    digits = "".join(filter(str.isdigit, img_path.stem))
    return int(digits) if digits else 0


# ── tiling ────────────────────────────────────────────────────────────────────

def generate_tiles(img: Image.Image, tile_size: int, overlap: float):
    """Yield (tile_crop, x_offset, y_offset) for all overlapping tiles."""
    W, H = img.size
    stride = max(1, int(tile_size * (1 - overlap)))

    def positions(length):
        pts = list(range(0, max(1, length - tile_size), stride))
        pts.append(max(0, length - tile_size))
        return sorted(set(pts))

    for y0 in positions(H):
        for x0 in positions(W):
            yield img.crop((x0, y0, min(x0 + tile_size, W), min(y0 + tile_size, H))), x0, y0


# ── tiled YOLO inference ──────────────────────────────────────────────────────

def run_tiled_inference(
    image_paths: list[Path],
    model: YOLO,
    conf: float = 0.15,
    iou: float = 0.45,
    tile_size: int = 640,
    overlap: float = 0.2,
    merge_iou: float = 0.5,
) -> dict[int, list[dict]]:
    """Return {image_id: [detections]} with NMS applied across tiles."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_by_image: dict[int, list[dict]] = {}

    for img_path in image_paths:
        img_id = image_id_from_path(img_path)
        img = Image.open(img_path).convert("RGB")

        all_boxes, all_scores, all_classes = [], [], []

        for tile, x0, y0 in generate_tiles(img, tile_size, overlap):
            preds = model.predict(source=tile, conf=conf, iou=iou, verbose=False, device=device)
            for pred in preds:
                if pred.boxes is None:
                    continue
                for box in pred.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    all_boxes.append([x0 + x1, y0 + y1, x0 + x2, y0 + y2])
                    all_scores.append(float(box.conf[0].item()))
                    all_classes.append(int(box.cls[0].item()))

        if not all_boxes:
            results_by_image[img_id] = []
            continue

        keep = batched_nms(
            torch.tensor(all_boxes, dtype=torch.float32),
            torch.tensor(all_scores, dtype=torch.float32),
            torch.tensor(all_classes, dtype=torch.int64),
            iou_threshold=merge_iou,
        )

        dets = []
        for i in keep.tolist():
            x1, y1, x2, y2 = all_boxes[i]
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": all_classes[i],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(all_scores[i], 4),
                }
            )
        results_by_image[img_id] = dets

    return results_by_image


# ── reference re-ranking ──────────────────────────────────────────────────────

def load_reference_data(device: str):
    """Return (feature_model, ref_matrix, ref_labels) or (None, None, None)."""
    emb_path = Path("reference_embeddings.npy")
    lbl_path = Path("reference_labels.json")
    mdl_path = Path("feature_extractor.pt")

    if not (emb_path.exists() and lbl_path.exists() and mdl_path.exists()):
        return None, None, None

    try:
        import torchvision.models as tvm
        import torchvision.transforms.functional as TF  # noqa: F401 (import check)

        ref_matrix = torch.tensor(np.load(str(emb_path)), dtype=torch.float32).to(device)
        with open(str(lbl_path)) as f:
            ref_labels = json.load(f)

        # Rebuild MobileNetV3-small without classifier head
        backbone = tvm.mobilenet_v3_small()
        backbone.classifier = torch.nn.Identity()
        state = torch.load(str(mdl_path), map_location=device)
        backbone.load_state_dict(state)
        backbone.eval().to(device)

        print(f"Reference re-ranking enabled: {len(ref_labels)} categories")
        return backbone, ref_matrix, ref_labels
    except Exception as e:
        print(f"Reference re-ranking disabled ({e})")
        return None, None, None


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_crops(crops: list[Image.Image], device: str) -> torch.Tensor:
    tensors = []
    for crop in crops:
        arr = np.array(crop.resize((224, 224)), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 224, 224)
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(t)
    return torch.stack(tensors).to(device)


def rerank_detections(
    img_path: Path,
    detections: list[dict],
    feature_model,
    ref_matrix: torch.Tensor,
    ref_labels: list[int],
    threshold: float = 0.65,
    device: str = "cpu",
) -> list[dict]:
    if not detections or feature_model is None:
        return detections

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    crops = []
    for det in detections:
        x, y, w, h = det["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        crops.append(img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img.crop((0, 0, 10, 10)))

    batch = preprocess_crops(crops, device)
    with torch.no_grad():
        feats = feature_model(batch)
    feats = F.normalize(feats, dim=1)                     # (N, D)
    sims = feats @ ref_matrix.T                            # (N, num_refs)
    best_sim, best_idx = sims.max(dim=1)

    reranked = []
    for i, det in enumerate(detections):
        det = dict(det)
        if best_sim[i].item() >= threshold:
            det["category_id"] = ref_labels[best_idx[i].item()]
        reranked.append(det)
    return reranked


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NorgesGruppen object detection inference")
    parser.add_argument("--input",     required=True,  help="Directory with input images")
    parser.add_argument("--output",    required=True,  help="Path to write predictions JSON")
    parser.add_argument("--conf",      type=float, default=0.15,  help="Detection confidence threshold")
    parser.add_argument("--tile-size", type=int,   default=640,   help="Tile size for sliced inference")
    parser.add_argument("--overlap",   type=float, default=0.2,   help="Tile overlap ratio")
    parser.add_argument("--no-tile",   action="store_true",       help="Disable tiled inference")
    parser.add_argument("--rerank-threshold", type=float, default=0.65,
                        help="Cosine similarity threshold for reference re-ranking")
    args = parser.parse_args()

    image_paths = collect_images([args.input])
    if not image_paths:
        print(f"WARNING: No images found in {args.input}")
        predictions = []
    else:
        print(f"Found {len(image_paths)} images")
        weights = find_model_weights()
        print(f"Loading YOLO model from {weights}")
        model = YOLO(weights)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if args.no_tile:
            # Standard full-image inference
            predictions = []
            for img_path in image_paths:
                img_id = image_id_from_path(img_path)
                preds = model.predict(source=str(img_path), conf=args.conf, iou=0.45,
                                      verbose=False, device=device)
                for pred in preds:
                    if pred.boxes is None:
                        continue
                    for box in pred.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        predictions.append({
                            "image_id": img_id,
                            "category_id": int(box.cls[0].item()),
                            "bbox": [round(x1,2), round(y1,2), round(x2-x1,2), round(y2-y1,2)],
                            "score": round(float(box.conf[0].item()), 4),
                        })
            results_by_image = None  # re-ranking not wired for no-tile path here
        else:
            print(f"Tiled inference (tile={args.tile_size}, overlap={args.overlap})")
            results_by_image = run_tiled_inference(
                image_paths, model,
                conf=args.conf,
                tile_size=args.tile_size,
                overlap=args.overlap,
            )

        # Reference re-ranking
        if not args.no_tile and results_by_image is not None:
            feature_model, ref_matrix, ref_labels = load_reference_data(device)
            predictions = []
            for img_path in image_paths:
                img_id = image_id_from_path(img_path)
                dets = results_by_image.get(img_id, [])
                if feature_model is not None:
                    dets = rerank_detections(
                        img_path, dets, feature_model, ref_matrix, ref_labels,
                        threshold=args.rerank_threshold, device=device,
                    )
                predictions.extend(dets)

        print(f"Total detections: {len(predictions)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, indent=2))
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
