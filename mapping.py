import cv2
import numpy as np
import os
import json
from flask import Flask, render_template, Response, request
from ultralytics import YOLO
from datetime import datetime

# ------------------ CONFIG ------------------
MODEL_PATH = "best_2D.pt"
CLASS_NAMES = ['fire', 'smoke', 'person']

MAP_W, MAP_H = 500, 500
OUTPUT_IMG_DIR = "outputs/images"
OUTPUT_MAP_DIR = "outputs/maps"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MAP_DIR, exist_ok=True)

# -------------------------------------------

app = Flask(__name__)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

map_img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

# ---------- Utility Functions ----------

def map_position(x, y, fw, fh):
    return int((x / fw) * MAP_W), int((y / fh) * MAP_H)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Camera Stream ----------

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO inference
        results = model(frame, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = CLASS_NAMES[cls_id]

                # Color by class
                if class_name == "fire":
                    color = (0, 0, 255)        # Red
                elif class_name == "smoke":
                    color = (160, 160, 160)    # Gray
                else:
                    color = (0, 255, 0)        # Green (person)

                label = f"{class_name} {conf:.2f}"

                # Draw box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # Encode frame for browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------- Web Routes ----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_object():
    target_class = request.form.get('object')

    success, frame = cap.read()
    if not success:
        return "Camera Error", 500

    results = model(frame, conf=0.4, verbose=False)
    h, w, _ = frame.shape

    saved = False
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id]

            if cls_name != target_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            mx, my = map_position(cx, cy, w, h)

            # Crop object
            crop = frame[y1:y2, x1:x2]

            # Save image
            img_name = f"{cls_name}_{timestamp()}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), crop)

            # Draw on map
            color = (0, 0, 255) if cls_name == 'fire' else \
                    (200, 200, 200) if cls_name == 'smoke' else \
                    (255, 255, 255)

            cv2.circle(map_img, (mx, my), 6, color, -1)

            detections.append({
                "class": cls_name,
                "map_x": mx,
                "map_y": my,
                "image": img_name
            })

            saved = True

    if saved:
        map_name = f"map_{timestamp()}.png"
        json_name = f"map_{timestamp()}.json"

        cv2.imwrite(os.path.join(OUTPUT_MAP_DIR, map_name), map_img)
        with open(os.path.join(OUTPUT_MAP_DIR, json_name), "w") as f:
            json.dump(detections, f, indent=2)

        return f"{target_class} captured and mapped!"

    return f"No {target_class} detected"

# ---------- Main ----------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
