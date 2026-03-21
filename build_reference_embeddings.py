"""
Build reference embeddings for classification re-ranking.

Run this LOCALLY (before zipping for submission) after you have:
  - data/coco/annotations.json  (train split)
  - data/product_images/        (extracted NM_NGD_product_images.zip)

Usage:
    python build_reference_embeddings.py \
        --coco  ~/Downloads/train \
        --refs  ~/Downloads/NM_NGD_product_images

Outputs (include all three in your submission zip):
    reference_embeddings.npy   — (N, D) float32 embedding matrix
    reference_labels.json      — list of N category_ids
    feature_extractor.pt       — MobileNetV3-small state dict
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image


VIEWS = ["main", "front", "back", "left", "right", "top", "bottom"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]


def load_feature_extractor(device: str):
    model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    return model


def embed_image(img_path: Path, model, device: str):
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            feat = model(t)
        return feat.squeeze(0).cpu()
    except Exception as e:
        print(f"    skip {img_path.name}: {e}")
        return None


def normalise_name(s: str) -> str:
    return s.lower().strip()


def build_barcode_to_catid(coco: dict, refs_dir: Path) -> dict:
    """
    Try three strategies to map barcode folder → category_id:
      1. product_code field in annotations (ideal)
      2. metadata.json in refs_dir (barcode → product_name → category_name match)
      3. Direct folder-name match against category names (last resort)
    """
    # ── Strategy 1: product_code in annotations ───────────────────────────────
    code_to_catid: dict[str, int] = {}
    for ann in coco.get("annotations", []):
        code = ann.get("product_code")
        if code:
            code_to_catid[str(code)] = ann["category_id"]

    if code_to_catid:
        print(f"  Strategy 1: found {len(code_to_catid)} barcode→category mappings in annotations.json")
        return code_to_catid

    # ── Strategy 2: metadata.json in product images dir ──────────────────────
    meta_path = refs_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

        # Build category name → id lookup
        cat_name_to_id = {normalise_name(c["name"]): c["id"]
                          for c in coco["categories"]}

        # metadata may be a list of dicts, or a dict keyed by barcode (values may be dicts or ints)
        if isinstance(metadata, list):
            items = [(None, item) for item in metadata]
        else:
            items = list(metadata.items())

        for key, val in items:
            if isinstance(val, dict):
                barcode  = str(val.get("barcode") or val.get("product_code") or val.get("gtin") or key or "")
                raw_name = val.get("product_name") or val.get("name", "")
                cat_id   = cat_name_to_id.get(normalise_name(raw_name))
                if barcode and cat_id is not None:
                    code_to_catid[barcode] = cat_id
            # If val is int/str (e.g. {barcode: count}), no product name available — skip

        if code_to_catid:
            print(f"  Strategy 2: matched {len(code_to_catid)} barcodes via metadata.json")
            return code_to_catid
        print("  Strategy 2: metadata.json found but values have no product names — trying strategy 3")
    else:
        print("  Strategy 2: no metadata.json found — trying strategy 3")

    # ── Strategy 3: substring match of barcode against category names ─────────
    cat_name_to_id = {normalise_name(c["name"]): c["id"]
                      for c in coco["categories"]}
    for folder in refs_dir.iterdir():
        if not folder.is_dir():
            continue
        barcode = folder.name
        for cat_name, cat_id in cat_name_to_id.items():
            if barcode in cat_name or cat_name in barcode:
                code_to_catid[barcode] = cat_id
                break

    print(f"  Strategy 3: matched {len(code_to_catid)} barcodes via name substring")
    return code_to_catid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco",       default="data/coco",           help="Folder with annotations.json")
    parser.add_argument("--refs",       default="data/product_images", help="Product reference images dir")
    parser.add_argument("--output-emb", default="reference_embeddings.npy")
    parser.add_argument("--output-lbl", default="reference_labels.json")
    parser.add_argument("--output-mdl", default="feature_extractor.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load annotations ──────────────────────────────────────────────────────
    ann_path = Path(args.coco) / "annotations.json"
    print(f"Loading {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)
    print(f"  {len(coco['categories'])} categories, {len(coco.get('annotations', []))} annotations")

    refs_dir = Path(args.refs)
    barcode_to_catid = build_barcode_to_catid(coco, refs_dir)

    if not barcode_to_catid:
        print("\nWARNING: Could not map any barcodes to category IDs.")
        print("Building embeddings for ALL product folders (no category label).")
        print("Re-ranking will be disabled at inference time.\n")
        # Save empty files so run.py doesn't crash
        np.save(args.output_emb, np.zeros((0, 576), dtype=np.float32))
        with open(args.output_lbl, "w") as f:
            json.dump([], f)
        print("Saved empty embedding files.")
        return

    # ── Load feature extractor ────────────────────────────────────────────────
    print("Loading MobileNetV3-small ...")
    feature_model = load_feature_extractor(device)
    torch.save(feature_model.state_dict(), args.output_mdl)
    print(f"Feature extractor saved → {args.output_mdl}")

    # ── Compute per-category average embeddings ───────────────────────────────
    catid_to_embeddings: dict[int, list] = defaultdict(list)

    for barcode, cat_id in barcode_to_catid.items():
        product_dir = refs_dir / barcode
        if not product_dir.exists():
            continue
        for view in VIEWS:
            for ext in IMG_EXTS:
                img_path = product_dir / f"{view}{ext}"
                if img_path.exists():
                    emb = embed_image(img_path, feature_model, device)
                    if emb is not None:
                        catid_to_embeddings[cat_id].append(emb)
                    break

    # ── Average + L2-normalise per category ──────────────────────────────────
    embeddings, labels = [], []
    for cat_id, embs in sorted(catid_to_embeddings.items()):
        avg = torch.stack(embs).mean(dim=0)
        avg = avg / (avg.norm() + 1e-8)
        embeddings.append(avg.numpy())
        labels.append(cat_id)

    if not embeddings:
        print("ERROR: no embeddings built — check that barcode folders exist in refs dir")
        return

    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)
    np.save(args.output_emb, emb_matrix)
    with open(args.output_lbl, "w") as f:
        json.dump(labels, f)

    print(f"\nBuilt embeddings for {len(labels)} categories")
    print(f"Saved {args.output_emb}  shape={emb_matrix.shape}")
    print(f"Saved {args.output_lbl}")
    print(f"\nInclude in submission zip:")
    print(f"  {args.output_emb}  {args.output_lbl}  {args.output_mdl}")


if __name__ == "__main__":
    main()
