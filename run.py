import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def letterbox(img, size=640):
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
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr, scale, pad_x, pad_y, orig_w, orig_h


def nms(boxes, scores, iou_threshold=0.45):
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
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Input: {args.input}", flush=True)
    print(f"Output: {args.output}", flush=True)

    model_path = Path(__file__).parent / "best.onnx"
    print(f"Model: {model_path} (exists={model_path.exists()})", flush=True)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print(f"ONNX providers: {session.get_providers()}", flush=True)

    input_path = Path(args.input)
    images = sorted([f for f in input_path.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"Found {len(images)} images", flush=True)

    predictions = []
    for img_path in images:
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(str(img_path)).convert("RGB")
        inp, scale, pad_x, pad_y, orig_w, orig_h = letterbox(img)

        outputs = session.run(None, {input_name: inp})
        pred = outputs[0][0].T  # [8400, 4+nc]

        class_scores = pred[:, 4:]
        best_class = class_scores.argmax(axis=1)
        best_score = class_scores.max(axis=1)

        mask = best_score >= args.conf
        pred = pred[mask]
        best_class = best_class[mask]
        best_score = best_score[mask]

        if len(pred) == 0:
            continue

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = nms(boxes, best_score)
        boxes = boxes[keep]
        best_class = best_class[keep]
        best_score = best_score[keep]

        for i in range(len(boxes)):
            rx1 = max(0.0, (boxes[i, 0] - pad_x) / scale)
            ry1 = max(0.0, (boxes[i, 1] - pad_y) / scale)
            rx2 = min(float(orig_w), (boxes[i, 2] - pad_x) / scale)
            ry2 = min(float(orig_h), (boxes[i, 3] - pad_y) / scale)
            bw = rx2 - rx1
            bh = ry2 - ry1
            if bw <= 0 or bh <= 0:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(best_class[i]),
                "bbox": [round(rx1, 1), round(ry1, 1), round(bw, 1), round(bh, 1)],
                "score": round(float(best_score[i]), 3),
            })

    print(f"Total detections: {len(predictions)}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
