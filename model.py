import io
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

CONFIDENCE_THRESHOLD = 0.5


def predict(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    results = model(img, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "confidence": round(conf, 4),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
        })

    has_tick = len(detections) > 0
    max_conf = max((d["confidence"] for d in detections), default=0.0)

    return {
        "has_tick": has_tick,
        "confidence": max_conf,
        "image_size": [w, h],
        "detections": detections,
    }
