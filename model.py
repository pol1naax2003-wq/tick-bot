import io
import gc
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.5
_session = None


def get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        logger.info("Loading ONNX model...")
        _session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
        logger.info("Model loaded!")
    return _session


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

        output = outputs[0][0]
        detections = []

        for i in range(output.shape[1]):
            cx, cy, bw, bh = output[0, i], output[1, i], output[2, i], output[3, i]
            conf = float(output[4, i])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            x2 = int((cx + bw / 2) * w / 640)
            y2 = int((cy + bh / 2) * h / 640)
            detections.append({
                "confidence": round(conf, 4),
                "bbox": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
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
