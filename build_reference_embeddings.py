"""
Build reference embeddings for classification re-ranking.

Run this LOCALLY (before zipping for submission) after you have:
  - best.pt   (your trained YOLO model)
  - data/coco/annotations.json
  - data/product_images/  (extracted NM_NGD_product_images.zip)

Usage:
    python build_reference_embeddings.py \
        --coco data/coco \
        --refs data/product_images \
        --weights best.pt

Outputs (include all three in your submission zip):
    reference_embeddings.npy   — (N, D) float32 embedding matrix
    reference_labels.json      — list of N category_ids
    feature_extractor.pt       — MobileNetV3-small state dict
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image


VIEWS = ["main", "front", "back", "left", "right", "top", "bottom"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]


def load_feature_extractor(device: str):
    """Load MobileNetV3-small without the classifier head."""
    model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    return model


def embed_image(img_path: Path, model, device: str) -> torch.Tensor | None:
    """Return a 1-D embedding tensor or None on failure."""
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,224,224)
        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            feat = model(t)  # (1, D)
        return feat.squeeze(0).cpu()
    except Exception as e:
        print(f"    skip {img_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco",    default="data/coco",          help="COCO dataset dir")
    parser.add_argument("--refs",    default="data/product_images", help="Product reference images dir")
    parser.add_argument("--weights", default="best.pt",            help="YOLO weights (unused here)")
    parser.add_argument("--output-emb",  default="reference_embeddings.npy")
    parser.add_argument("--output-lbl",  default="reference_labels.json")
    parser.add_argument("--output-mdl",  default="feature_extractor.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── 1. Parse annotations to build category_id → product_codes mapping ──────
    ann_path = Path(args.coco) / "annotations.json"
    print(f"Loading {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_to_codes: dict[int, set] = defaultdict(set)
    for ann in coco["annotations"]:
        code = ann.get("product_code")
        if code:
            cat_to_codes[ann["category_id"]].add(str(code))

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    print(f"  {len(categories)} categories, {len(cat_to_codes)} with product codes")

    # ── 2. Load feature extractor ─────────────────────────────────────────────
    print("Loading MobileNetV3-small ...")
    feature_model = load_feature_extractor(device)

    # Save state dict (classifier replaced with Identity, so state_dict is same)
    torch.save(feature_model.state_dict(), args.output_mdl)
    print(f"Feature extractor saved to {args.output_mdl}")

    # ── 3. Compute reference embeddings ───────────────────────────────────────
    refs_dir = Path(args.refs)
    embeddings = []
    labels = []
    skipped = 0

    for cat_id, codes in sorted(cat_to_codes.items()):
        cat_embeddings = []
        for code in codes:
            product_dir = refs_dir / code
            if not product_dir.exists():
                continue
            for view in VIEWS:
                for ext in IMG_EXTS:
                    img_path = product_dir / f"{view}{ext}"
                    if img_path.exists():
                        emb = embed_image(img_path, feature_model, device)
                        if emb is not None:
                            cat_embeddings.append(emb)
                        break  # only first matching ext per view

        if not cat_embeddings:
            skipped += 1
            continue

        avg = torch.stack(cat_embeddings).mean(dim=0)
        avg = avg / (avg.norm() + 1e-8)  # L2 normalise
        embeddings.append(avg.numpy())
        labels.append(cat_id)

        if len(labels) % 50 == 0:
            print(f"  Processed {len(labels)} categories ...")

    print(f"\nBuilt embeddings for {len(labels)} categories ({skipped} skipped — no reference images)")

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, D)
    np.save(args.output_emb, emb_matrix)
    with open(args.output_lbl, "w") as f:
        json.dump(labels, f)

    print(f"Saved {args.output_emb}  shape={emb_matrix.shape}")
    print(f"Saved {args.output_lbl}  ({len(labels)} entries)")
    print(f"\nInclude these files in your submission zip:")
    print(f"  {args.output_emb}")
    print(f"  {args.output_lbl}")
    print(f"  {args.output_mdl}")


if __name__ == "__main__":
    main()
