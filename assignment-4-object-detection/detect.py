"""Live webcam detection: YOLOv8 finds the objects (boxes), MobileNetV2 names what is inside each box.

Press 'q' to quit.
"""
import time

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

CONF = 0.35          # YOLO minimum confidence: lower = more boxes, higher = fewer wrong boxes
MAX_CLASSIFY = 5     # run MobileNet on at most this many boxes per frame (keeps the video smooth)

# 1. Models
yolo_model = YOLO("yolov8n.pt")                   # detects 80 everyday object types and draws boxes
mobilenet_model = MobileNetV2(weights="imagenet")  # names 1000 object types from a cropped image


def classify_crop(crop):
    """MobileNetV2 prediction for ONE cropped object (not the whole frame)."""
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
    x = preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))
    preds = mobilenet_model(x, training=False).numpy()
    _, name, score = decode_predictions(preds, top=1)[0][0]
    return name.replace("_", " "), float(score)


def draw_box(frame, x1, y1, x2, y2, label, color):
    """Thick rectangle plus a label on a filled background so it is easy to read."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = y1 - th - 8 if y1 - th - 8 > 0 else y1        # put the label inside if the box is at the top edge
    left = max(0, min(x1, frame.shape[1] - tw - 6))     # keep the label inside the picture on the right
    cv2.rectangle(frame, (left, top), (left + tw + 6, top + th + 8), color, -1)
    cv2.putText(frame, label, (left + 3, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


# 2. Open the webcam (CAP_DSHOW is more reliable on Windows; plain open is the fallback)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    raise SystemExit

print("Press 'q' on your keyboard to exit the video loop.")
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # --- YOLO: find objects and their boxes ---
    results = yolo_model(frame, conf=CONF, verbose=False)[0]
    h, w = frame.shape[:2]
    boxes = sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)   # keep the box inside the picture
        class_id = int(box.cls[0])
        label = f"{results.names[class_id]} {float(box.conf[0]):.0%}"

        # --- MobileNet: name what is inside this box (only the most confident boxes) ---
        if i < MAX_CLASSIFY and (x2 - x1) > 20 and (y2 - y1) > 20:
            name, score = classify_crop(frame[y1:y2, x1:x2])
            label += f" | MobileNet: {name} {score:.0%}"

        color = ((class_id * 67) % 140 + 115, (class_id * 131) % 140 + 115, (class_id * 29) % 140 + 115)   # bright, so black text is readable
        draw_box(frame, x1, y1, x2, y2, label, color)

    # speed and object count in the corner
    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"Objects: {len(boxes)}   FPS: {fps:.1f}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("AI Image Recognition (YOLO + MobileNet)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 3. Clean up
cap.release()
cv2.destroyAllWindows()
