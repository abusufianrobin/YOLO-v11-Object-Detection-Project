# burn_detect.py
import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------
# Command line arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (example: "best.pt")')
parser.add_argument('--source', required=True, help='Source: "usb0" for USB camera, video file, image, or folder')
parser.add_argument('--thresh', type=float, default=0.4, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Display resolution WxH (e.g., "1280x720")')
parser.add_argument('--record', action='store_true', help='Save output video')
parser.add_argument('--distance', type=float, default=50.0, help='Distance to object in cm')
parser.add_argument('--focal', type=float, default=800.0, help='Camera focal length in pixels')
args = parser.parse_args()

# ---------------------------
# Load YOLO model
# ---------------------------
if not os.path.exists(args.model):
    print(f"ERROR: Model file '{args.model}' not found!")
    sys.exit(1)

model = YOLO(args.model, task='detect')
labels = model.names

# ---------------------------
# Parse source
# ---------------------------
img_ext = ['.jpg','.jpeg','.png','.bmp']
vid_ext = ['.mp4','.avi','.mov','.mkv','.wmv']

source_type = None
if os.path.isdir(args.source):
    source_type = 'folder'
elif os.path.isfile(args.source):
    ext = os.path.splitext(args.source)[1].lower()
    if ext in img_ext: source_type = 'image'
    elif ext in vid_ext: source_type = 'video'
    else:
        print(f"File type {ext} not supported!")
        sys.exit(1)
elif 'usb' in args.source:
    source_type = 'usb'
    usb_idx = int(args.source[3:])
else:
    print(f"Invalid source: {args.source}")
    sys.exit(1)

# ---------------------------
# Resolution
# ---------------------------
resize = False
if args.resolution:
    resize = True
    resW, resH = map(int, args.resolution.split('x'))

# ---------------------------
# Recording setup
# ---------------------------
if args.record:
    if source_type not in ['video','usb']:
        print("Recording only works for video or camera sources.")
        sys.exit(1)
    if not resize:
        print("Please specify resolution when recording!")
        sys.exit(1)
    recorder = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

# ---------------------------
# Initialize camera/source
# ---------------------------
if source_type == 'image':
    imgs_list = [args.source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(args.source,'*')) if os.path.splitext(f)[1].lower() in img_ext]
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)
    if resize:
        cap.set(3,resW)
        cap.set(4,resH)
elif source_type == 'video':
    cap = cv2.VideoCapture(args.source)
    if resize:
        cap.set(3,resW)
        cap.set(4,resH)

# ---------------------------
# Bounding box colors
# ---------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# ---------------------------
# Inference loop
# ---------------------------
img_count = 0
fps_buffer = []
fps_avg_len = 200

while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            print("All images processed. Exiting...")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video','usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0

    for i, det in enumerate(detections):
        conf = det.conf.item()
        if conf < args.thresh: 
            continue

        xmin, ymin, xmax, ymax = det.xyxy[0].cpu().numpy().astype(int)
        class_id = int(det.cls.item())
        label = labels[class_id]
        color = bbox_colors[class_id % len(bbox_colors)]

        # Draw bounding box
        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,2)
        label_text = f"{label}: {conf:.2f}"
        cv2.putText(frame,label_text,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        # Compute width & height in cm
        w_px, h_px = xmax-xmin, ymax-ymin
        w_cm = w_px * args.distance / args.focal
        h_cm = h_px * args.distance / args.focal
        cv2.putText(frame, f"W:{w_cm:.1f}cm H:{h_cm:.1f}cm", (xmin, ymax+15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        object_count += 1

    # FPS
    fps = 1.0 / (time.perf_counter() - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)
    cv2.putText(frame,f'FPS:{avg_fps:.2f}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame,f'Objects:{object_count}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    # Show frame
    cv2.imshow("YOLO Detection", frame)
    if args.record:
        recorder.write(frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'): # save snapshot
        cv2.imwrite(f'capture_{int(time.time())}.png', frame)

# ---------------------------
# Cleanup
# ---------------------------
if source_type in ['video','usb']:
    cap.release()
if args.record:
    recorder.release()
cv2.destroyAllWindows()
print(f"Average FPS: {avg_fps:.2f}")
