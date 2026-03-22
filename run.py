"""
NorgesGruppen Object Detection — submission entry point.

python run.py --input /data/images --output /output/predictions.json

Security-safe: no os/sys/subprocess/pickle/yaml/requests imports.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.models as tvm

def weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                          iou_thr=0.55, skip_box_thr=0.0):
    """
    Native WBF implementation — no external package needed.
    boxes in [0,1] format [x1,y1,x2,y2].
    Score of each cluster = sum(scores) / n_models, so boxes seen in
    more views get higher relative score (proper ensemble behaviour).
    """
    n_models = len(boxes_list)
    if n_models == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    flat_b, flat_s, flat_l = [], [], []
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for b, s, l in zip(boxes, scores, labels):
            if s >= skip_box_thr:
                flat_b.append(np.asarray(b, dtype=np.float32))
                flat_s.append(float(s))
                flat_l.append(int(l))

    if not flat_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    flat_b = np.array(flat_b)
    flat_s = np.array(flat_s)
    flat_l = np.array(flat_l, dtype=int)
    order  = np.argsort(-flat_s)
    flat_b, flat_s, flat_l = flat_b[order], flat_s[order], flat_l[order]

    c_boxes, c_scores, c_labels, c_avg = [], [], [], []

    for b, s, l in zip(flat_b, flat_s, flat_l):
        best_ci, best_iou = -1, iou_thr
        for ci, avg_b in enumerate(c_avg):
            if c_labels[ci][int(np.argmax(c_scores[ci]))] != l:
                continue
            ix1 = max(b[0], avg_b[0]); iy1 = max(b[1], avg_b[1])
            ix2 = min(b[2], avg_b[2]); iy2 = min(b[3], avg_b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = ((b[2]-b[0])*(b[3]-b[1]) +
                     (avg_b[2]-avg_b[0])*(avg_b[3]-avg_b[1]) - inter)
            iou_v = inter / union if union > 0 else 0.0
            if iou_v > best_iou:
                best_iou, best_ci = iou_v, ci

        if best_ci >= 0:
            c_boxes[best_ci].append(b)
            c_scores[best_ci].append(s)
            c_labels[best_ci].append(l)
            ws = np.array(c_scores[best_ci])
            c_avg[best_ci] = np.average(np.array(c_boxes[best_ci]), axis=0, weights=ws)
        else:
            c_boxes.append([b]); c_scores.append([s])
            c_labels.append([l]); c_avg.append(b.copy())

    out_b, out_s, out_l = [], [], []
    for cb, cs, cl, avg_b in zip(c_boxes, c_scores, c_labels, c_avg):
        final_score = max(cs)
        if final_score < skip_box_thr:
            continue
        lab_s = {}
        for ll, ss in zip(cl, cs):
            lab_s[ll] = lab_s.get(ll, 0) + ss
        out_b.append(avg_b)
        out_s.append(final_score)
        out_l.append(max(lab_s, key=lab_s.get))

    if not out_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.array(out_b), np.array(out_s), np.array(out_l, dtype=int)


WBF_AVAILABLE = True


# ── Image preprocessing ───────────────────────────────────────────────────────

def letterbox(img: Image.Image, size: int = 640):
    """Resize + pad image to square, keeping aspect ratio."""
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
    return arr, scale, pad_x, pad_y, orig_w, orig_h


def hflip(arr: np.ndarray) -> np.ndarray:
    """Horizontal-flip a NCHW array."""
    return arr[:, :, :, ::-1].copy()


# ── NMS fallback ──────────────────────────────────────────────────────────────

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


# ── ONNX inference ────────────────────────────────────────────────────────────

def run_session(session, input_name: str, arr: np.ndarray) -> np.ndarray:
    """Return raw predictions [N, 4+nc]."""
    return session.run(None, {input_name: arr})[0][0].T


def decode_preds(pred: np.ndarray, conf_thr: float):
    """Filter predictions by confidence, return (boxes_xyxy, scores, classes)."""
    class_scores = pred[:, 4:]
    best_class = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)
    mask = best_score >= conf_thr
    pred = pred[mask]
    best_class = best_class[mask]
    best_score = best_score[mask]
    if len(pred) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1), best_score, best_class


def infer_region(session, input_name, region: Image.Image, INPUT_SIZE, do_tta, conf_thr):
    """
    Letterbox region to INPUT_SIZE, run YOLO (+TTA), return list of
    (boxes_xyxy, scores, classes) — one entry per inference view so that
    WBF can properly fuse them as separate models.
    """
    inp, scale, pad_x, pad_y, orig_w, orig_h = letterbox(region, INPUT_SIZE)
    view_preds = [run_session(session, input_name, inp)]
    if do_tta:
        pred_flip = run_session(session, input_name, hflip(inp))
        pred_flip[:, 0] = INPUT_SIZE - pred_flip[:, 0]  # un-flip cx
        view_preds.append(pred_flip)

    results = []
    for pred in view_preds:
        boxes_lb, scores, classes = decode_preds(pred, conf_thr)
        if len(boxes_lb) == 0:
            results.append((np.zeros((0, 4)), scores, classes))
            continue
        x1 = np.clip((boxes_lb[:, 0] - pad_x) / scale, 0.0, orig_w)
        y1 = np.clip((boxes_lb[:, 1] - pad_y) / scale, 0.0, orig_h)
        x2 = np.clip((boxes_lb[:, 2] - pad_x) / scale, 0.0, orig_w)
        y2 = np.clip((boxes_lb[:, 3] - pad_y) / scale, 0.0, orig_h)
        results.append((np.stack([x1, y1, x2, y2], axis=1), scores, classes))
    return results


def get_tile_offsets(total: int, tile: int, stride: int) -> list:
    """Return start offsets so tiles cover [0, total] completely."""
    offsets = []
    x = 0
    while True:
        offsets.append(x)
        if x + tile >= total:
            break
        x = min(x + stride, total - tile)
    return offsets


# ── Reference-embedding re-ranker ─────────────────────────────────────────────

def load_classifier(cls_path: Path, device: str):
    model = tvm.mobilenet_v3_small()
    model.classifier = torch.nn.Identity()
    model.load_state_dict(torch.load(str(cls_path), map_location=device))
    model.eval().to(device)
    return model


def embed_crop(crop: Image.Image, classifier, device: str) -> np.ndarray:
    if crop.width < 4 or crop.height < 4:
        return None
    arr = np.array(crop.resize((224, 224)), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        feat = classifier(t).squeeze(0)
    feat = feat / (feat.norm() + 1e-8)
    return feat.cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf",         type=float, default=0.15,
                        help="Minimum detection confidence")
    parser.add_argument("--iou",          type=float, default=0.45,
                        help="IoU threshold for WBF/NMS")
    parser.add_argument("--sim-thresh",   type=float, default=0.50,
                        help="Cosine similarity threshold for reference re-ranking")
    parser.add_argument("--tile-overlap", type=float, default=0.25,
                        help="Tile overlap fraction (0=no overlap, 0.25=25%%)")
    parser.add_argument("--no-tta",  action="store_true",
                        help="Disable test-time augmentation (horizontal flip)")
    parser.add_argument("--tile",    action="store_true",
                        help="Enable tiled inference (experimental)")
    args = parser.parse_args()

    print(f"Input:  {args.input}",  flush=True)
    print(f"Output: {args.output}", flush=True)

    root = Path(__file__).parent

    # ── Detector ──────────────────────────────────────────────────────────────
    model_path = root / "best.onnx"
    print(f"Detector: {model_path}  exists={model_path.exists()}", flush=True)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    INPUT_SIZE = session.get_inputs()[0].shape[2]  # auto-detect from model
    print(f"Active provider: {session.get_providers()[0]}  INPUT_SIZE={INPUT_SIZE}", flush=True)

    TILE_SIZE   = INPUT_SIZE
    TILE_STRIDE = max(1, int(TILE_SIZE * (1.0 - args.tile_overlap)))

    # ── Optional re-ranker ────────────────────────────────────────────────────
    emb_path = root / "reference_embeddings.npy"
    lbl_path = root / "reference_labels.json"
    cls_path = root / "feature_extractor.pt"
    use_rerank = emb_path.exists() and lbl_path.exists() and cls_path.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_rerank:
        print("Loading reference embeddings for re-ranking ...", flush=True)
        ref_embs = torch.from_numpy(np.load(str(emb_path))).to(device)
        with open(str(lbl_path)) as f:
            ref_labels = json.load(f)
        if len(ref_labels) == 0:
            use_rerank = False
            print("  Reference embeddings empty — using YOLO classification only", flush=True)
        else:
            classifier = load_classifier(cls_path, device)
            print(f"  {len(ref_labels)} reference categories loaded", flush=True)
    else:
        print("No reference embeddings — using YOLO classification only", flush=True)

    # ── Iterate images ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    images = sorted([f for f in input_path.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    do_tta  = not args.no_tta
    do_tile = args.tile
    print(f"Found {len(images)} images  tile={do_tile}  tta={do_tta}", flush=True)
    predictions = []

    for img_path in images:
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(str(img_path)).convert("RGB")
        W, H = img.width, img.height

        # Accumulate (boxes_norm, scores, classes) per inference view for WBF
        boxes_list  = []
        scores_list = []
        labels_list = []

        def _add(boxes, scores, classes):
            if len(boxes) == 0:
                return
            # Normalise to [0, 1] relative to full image
            bn = np.clip(boxes / np.array([W, H, W, H], dtype=np.float32), 0.0, 1.0)
            boxes_list.append(bn.tolist())
            scores_list.append(scores.tolist())
            labels_list.append(classes.tolist())

        # ── 1. Full-image inference (downsampled to INPUT_SIZE) ────────────
        for b, s, c in infer_region(session, input_name, img, INPUT_SIZE, do_tta, args.conf):
            _add(b, s, c)

        # ── 2. Tiled inference (native resolution for each tile) ───────────
        if do_tile and (W > TILE_SIZE or H > TILE_SIZE):
            x_offsets = get_tile_offsets(W, TILE_SIZE, TILE_STRIDE)
            y_offsets = get_tile_offsets(H, TILE_SIZE, TILE_STRIDE)
            print(f"  {img_path.name}: {W}x{H}  tiles={len(x_offsets)*len(y_offsets)}", flush=True)
            for tx in x_offsets:
                for ty in y_offsets:
                    tw = min(TILE_SIZE, W - tx)
                    th = min(TILE_SIZE, H - ty)
                    tile_img = img.crop((tx, ty, tx + tw, ty + th))
                    for tb, ts, tc in infer_region(session, input_name, tile_img,
                                                   INPUT_SIZE, do_tta, args.conf):
                        if len(tb) > 0:
                            tb[:, 0] += tx; tb[:, 2] += tx
                            tb[:, 1] += ty; tb[:, 3] += ty
                        _add(tb, ts, tc)

        if not boxes_list:
            continue

        # ── Merge with WBF (or NMS fallback) ──────────────────────────────
        if WBF_AVAILABLE:
            wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                iou_thr=args.iou,
                skip_box_thr=args.conf,
            )
            all_boxes   = np.array(wbf_boxes) * np.array([W, H, W, H])
            all_scores  = np.array(wbf_scores)
            all_classes = np.array(wbf_labels, dtype=int)
        else:
            all_boxes   = np.concatenate(
                [np.array(b) * np.array([W, H, W, H]) for b in boxes_list])
            all_scores  = np.concatenate([np.array(s) for s in scores_list])
            all_classes = np.concatenate([np.array(c, dtype=int) for c in labels_list])
            keep = nms(all_boxes, all_scores, args.iou)
            all_boxes   = all_boxes[keep]
            all_scores  = all_scores[keep]
            all_classes = all_classes[keep]

        # ── Convert to output format + optional re-ranking ─────────────────
        for i in range(len(all_boxes)):
            rx1, ry1, rx2, ry2 = all_boxes[i]
            bw = rx2 - rx1
            bh = ry2 - ry1
            if bw <= 0 or bh <= 0:
                continue

            cat_id = int(all_classes[i])
            score  = float(all_scores[i])

            # Skip 'unknown_product' class (355) — not a valid competition category
            if cat_id >= 355:
                continue

            if use_rerank and score >= 0.25:
                crop = img.crop((rx1, ry1, rx2, ry2))
                emb = embed_crop(crop, classifier, device)
                if emb is not None:
                    emb_t = torch.from_numpy(emb).to(device)
                    sims  = (ref_embs @ emb_t).cpu().numpy()
                    best_ref_idx = int(sims.argmax())
                    sim_val = float(sims[best_ref_idx])
                    if sim_val >= args.sim_thresh:
                        cat_id = int(ref_labels[best_ref_idx])
                        score  = round(0.7 * score + 0.3 * sim_val, 3)

            predictions.append({
                "image_id":    image_id,
                "category_id": cat_id,
                "bbox": [round(float(rx1), 1), round(float(ry1), 1),
                         round(bw, 1), round(bh, 1)],
                "score": round(score, 3),
            })

    print(f"Total detections: {len(predictions)}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
