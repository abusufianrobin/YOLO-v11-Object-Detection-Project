# ===================================================== 
# YOLO Object Detection + Bounding-Box–Guided 2D Object Scanning
# Meaningful Object-Centric 2D Spatial Mapping
# =====================================================

import os
import sys
import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --------------------------- Command line arguments ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file')
parser.add_argument('--source', required=True, help='usb0 | video | image | folder')
parser.add_argument('--thresh', type=float, default=0.4)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

# --------------------------- 2D MAP CONFIG ---------------------------
MAP_SIZE = 500
RESOLUTION = 0.05   # meters per cell
MAP_CENTER = MAP_SIZE // 2
MAX_DEPTH_METERS = 6.0
fx = fy = 800.0

occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

# --------------------------- Load YOLO ---------------------------
print(f"Loading YOLO model: {args.model}")
model = YOLO(args.model)
labels = model.names

# --------------------------- Load MiDaS Depth ---------------------------
print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = transforms.small_transform
print("MiDaS ready")

# --------------------------- Video Source ---------------------------
if 'usb' in args.source:
    cap = cv2.VideoCapture(int(args.source[3:]))
elif os.path.isfile(args.source):
    cap = cv2.VideoCapture(args.source)
else:
    print("Invalid source")
    sys.exit(1)

# --------------------------- Depth Estimation ---------------------------
def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(img)
    with torch.no_grad():
        depth = midas(inp)
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth

# --------------------------- Project bounding-box pixels ---------------------------
def scan_bbox(depth, bbox, frame_shape):
    fh, fw = frame_shape[:2]
    dh, dw = depth.shape
    x1, y1, x2, y2 = bbox
    cx = fw // 2
    color = tuple(int(c) for c in np.random.randint(80, 255, 3))
    for v in range(y1, y2, 6):
        for u in range(x1, x2, 6):
            du = int(u * dw / fw)
            dv = int(v * dh / fh)
            if du < 0 or du >= dw or dv < 0 or dv >= dh:
                continue
            z = depth[dv, du] * MAX_DEPTH_METERS
            if z < 0.5 or z > MAX_DEPTH_METERS:
                continue
            x = (u - cx) * z / fx
            mx = int(MAP_CENTER + x / RESOLUTION)
            my = int(MAP_CENTER + z / RESOLUTION)
            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                occupancy_map[my, mx] = color

# --------------------------- MAIN LOOP ---------------------------
fps_buffer = []

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    if args.resolution:
        w, h = map(int, args.resolution.split('x'))
        frame = cv2.resize(frame, (w, h))

    depth = estimate_depth(frame)
    results = model(frame, verbose=False)
    detections = results[0].boxes

    for det in detections:
        conf = det.conf.item()
        if conf < args.thresh:
            continue

        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
        cls = int(det.cls.item())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, labels[cls], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        scan_bbox(depth, (x1, y1, x2, y2), frame.shape)

    fps = 1.0 / (time.perf_counter() - t0)
    fps_buffer.append(fps)
    if len(fps_buffer) > 50:
        fps_buffer.pop(0)

    map_vis = cv2.flip(occupancy_map, 0)

    cv2.imshow("Camera", frame)
    cv2.imshow("2D Object Scan", map_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
