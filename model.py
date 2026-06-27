import io
import gc
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
_session = None


def get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        logger.info("Loading ONNX model...")
        _session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
        logger.info("Model loaded!")
    return _session


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def predict(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        img_resized = img.resize((640, 640))
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)

        session = get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: arr})

        output = outputs[0][0]  # shape [5, 8400]

        all_boxes = []
        all_scores = []

        for i in range(output.shape[1]):
            conf = float(output[4, i])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            cx, cy, bw, bh = output[0, i], output[1, i], output[2, i], output[3, i]
            x1 = (cx - bw / 2) * w / 640
            y1 = (cy - bh / 2) * h / 640
            x2 = (cx + bw / 2) * w / 640
            y2 = (cy + bh / 2) * h / 640
            all_boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
            all_scores.append(conf)

        detections = []
        if all_boxes:
            boxes = np.array(all_boxes)
            scores = np.array(all_scores)
            keep = nms(boxes, scores, IOU_THRESHOLD)
            for i in keep:
                detections.append({
                    "confidence": round(float(scores[i]), 4),
                    "bbox": [int(x) for x in boxes[i]],
                })

        del arr, output, outputs
        gc.collect()

        has_tick = len(detections) > 0
        max_conf = max((d["confidence"] for d in detections), default=0.0)

        return {
            "has_tick": has_tick,
            "confidence": max_conf,
            "image_size": [w, h],
            "detections": detections,
        }
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return {
            "has_tick": False,
            "confidence": 0,
            "image_size": [0, 0],
            "detections": [],
        }
