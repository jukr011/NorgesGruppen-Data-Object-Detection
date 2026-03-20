"""
Local training script for NorgesGruppen Object Detection.

Steps:
  1. Unzip NM_NGD_coco_dataset.zip → data/coco/
  2. Run:  python train.py --data data/coco --epochs 50
  3. Best weights saved to runs/detect/train/weights/best.pt
  4. Copy best.pt to repo root, zip, and upload.

Requirements:  pip install ultralytics
"""

import argparse
import json
import shutil
import yaml
from pathlib import Path

from ultralytics import YOLO


# ── COCO → YOLO conversion ────────────────────────────────────────────────────

def coco_to_yolo(coco_dir: Path, output_dir: Path, val_fraction: float = 0.1):
    """
    Convert COCO-format annotations to YOLO txt format.

    COCO bbox: [x, y, width, height]  (top-left corner, absolute pixels)
    YOLO bbox: [cx, cy, w, h]         (center, normalized 0-1)
    """
    ann_path = coco_dir / "annotations.json"
    img_dir = coco_dir / "images"

    print(f"Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookup tables
    images = {img["id"]: img for img in coco["images"]}
    num_classes = len(coco["categories"])
    # category id → 0-based index (COCO ids may not be contiguous)
    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    # Group annotations by image id
    from collections import defaultdict
    img_anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_anns[ann["image_id"]].append(ann)

    # Create output directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_ids = list(images.keys())
    split_idx = int(len(all_ids) * (1 - val_fraction))
    train_ids = set(all_ids[:split_idx])
    val_ids = set(all_ids[split_idx:])

    def write_split(ids, split_name):
        for img_id in ids:
            img_info = images[img_id]
            src_img = img_dir / img_info["file_name"]
            if not src_img.exists():
                continue

            W, H = img_info["width"], img_info["height"]
            dst_img = output_dir / "images" / split_name / img_info["file_name"]
            dst_lbl = output_dir / "labels" / split_name / (src_img.stem + ".txt")

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Write labels
            lines = []
            for ann in img_anns.get(img_id, []):
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                # Clamp to [0, 1]
                cx, cy, nw, nh = (
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                    max(0.0, min(1.0, nw)),
                    max(0.0, min(1.0, nh)),
                )
                cls_idx = cat_id_to_idx[ann["category_id"]]
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            dst_lbl.write_text("\n".join(lines))

        print(f"  {split_name}: {len(ids)} images")

    write_split(train_ids, "train")
    write_split(val_ids, "val")

    # Build class name list (in order)
    names = [cat["name"] for cat in coco["categories"]]

    return output_dir, names, num_classes


def write_dataset_yaml(yolo_dir: Path, names: list[str]) -> Path:
    cfg = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"Dataset YAML written to {yaml_path}")
    return yaml_path


# ── training ──────────────────────────────────────────────────────────────────

def train(
    data_yaml: Path,
    model_size: str = "m",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    workers: int = 4,
    resume: bool = False,
    patience: int = 50,
    cos_lr: bool = True,
):
    """Fine-tune YOLOv8 on the NorgesGruppen dataset."""
    model_name = f"yolov8{model_size}.pt"
    print(f"\nStarting training with {model_name}, {epochs} epochs, imgsz={imgsz}")

    model = YOLO(model_name)  # downloads pretrained weights from ultralytics if needed
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        resume=resume,
        patience=patience,
        project="runs/detect",
        name="train",
        # Learning rate schedule
        cos_lr=cos_lr,
        warmup_epochs=3,
        # Better classification with many classes
        label_smoothing=0.1,
        # Augmentation helpful for shelf images
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,  # disable mosaic in final 10 epochs for stable convergence
        # Save best checkpoint
        save=True,
        save_period=-1,  # only save best and last
    )

    best_weights = Path("runs/detect/train/weights/best.pt")
    if best_weights.exists():
        shutil.copy2(best_weights, "best.pt")
        print(f"\nBest weights copied to best.pt — include this file when zipping for submission.")
    return best_weights


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on NorgesGruppen COCO dataset")
    parser.add_argument(
        "--data", default="data/coco",
        help="Path to unzipped NM_NGD_coco_dataset folder (contains annotations.json + images/)"
    )
    parser.add_argument("--yolo-dir", default="data/yolo", help="Where to write YOLO-format data")
    parser.add_argument("--model", default="m", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience epochs")
    parser.add_argument("--no-cos-lr", action="store_true", help="Disable cosine LR schedule")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip COCO→YOLO conversion (use if data/yolo already exists)"
    )
    args = parser.parse_args()

    coco_dir = Path(args.data)
    yolo_dir = Path(args.yolo_dir)

    if not args.skip_convert:
        print("Converting COCO annotations to YOLO format...")
        yolo_dir, names, nc = coco_to_yolo(coco_dir, yolo_dir, args.val_fraction)
        print(f"  {nc} categories")
        data_yaml = write_dataset_yaml(yolo_dir, names)
    else:
        data_yaml = yolo_dir / "dataset.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Expected {data_yaml}. Run without --skip-convert first.")

    train(
        data_yaml=data_yaml,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        resume=args.resume,
        patience=args.patience,
        cos_lr=not args.no_cos_lr,
    )


if __name__ == "__main__":
    main()
