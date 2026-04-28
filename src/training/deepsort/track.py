"""
Sprint 05 — DeepSORT Object Tracking
Runs YOLOv8 detection + DeepSORT tracking on a sequence of COCO128 images.
Treats the sorted image list as consecutive frames of a scene.
Outputs per-frame tracking results (track_id, bbox, class) to JSON,
then uploads to Azure processed-silver.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from io import BytesIO

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# ── Config ────────────────────────────────────────────────────────────────────
STORAGE_ACCOUNT   = "trafficproddlake"
SILVER_CONTAINER  = "processed-silver"
MODEL_WEIGHTS     = Path(__file__).parents[2] / "training/yolov8/runs/yolov8n-traffic/weights/best.pt"
IMAGES_DIR        = Path(__file__).parents[3] / "data/coco128/images/train2017"
OUTPUT_DIR        = Path(__file__).parent / "output"
CONF_THRESHOLD    = 0.3
VEHICLE_CLASS_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck

COCO_NAMES = {
    2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}

# ── Setup ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
model   = YOLO(str(MODEL_WEIGHTS))
tracker = DeepSort(max_age=5, n_init=1, max_iou_distance=0.7)

image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
print(f"Processing {len(image_paths)} frames with YOLOv8 + DeepSORT...")

# ── Tracking loop ─────────────────────────────────────────────────────────────
all_tracks   = []
track_summary = {}   # track_id → {class, first_frame, last_frame, bbox_history}

for frame_idx, img_path in enumerate(image_paths):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    # YOLOv8 detection
    results  = model(img_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
    boxes    = results.boxes

    # Build detections list for DeepSORT: [[x1,y1,w,h], conf, class_id]
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=img_bgr)

    frame_tracks = []
    for track in tracks:
        if track.is_deleted():
            continue
        tid   = track.track_id
        ltrb  = track.to_ltrb()
        cls_id = track.det_class if track.det_class is not None else -1

        frame_tracks.append({
            "track_id": int(tid),
            "class":    COCO_NAMES.get(cls_id, "vehicle"),
            "bbox":     [round(float(v), 1) for v in ltrb],
        })

        # Update summary
        if tid not in track_summary:
            track_summary[tid] = {
                "class":        COCO_NAMES.get(cls_id, "vehicle"),
                "first_frame":  frame_idx,
                "last_frame":   frame_idx,
                "frame_count":  1,
            }
        else:
            track_summary[tid]["last_frame"] = frame_idx
            track_summary[tid]["frame_count"] += 1

    all_tracks.append({
        "frame": frame_idx,
        "image": img_path.name,
        "detections": len(detections),
        "tracks": frame_tracks,
    })

    if (frame_idx + 1) % 32 == 0:
        print(f"  Processed {frame_idx + 1}/{len(image_paths)} frames...")

print(f"Tracking complete. Unique vehicle tracks: {len(track_summary)}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "run_timestamp":    datetime.utcnow().isoformat(),
    "total_frames":     len(image_paths),
    "unique_tracks":    len(track_summary),
    "vehicle_classes":  list(COCO_NAMES.values()),
    "track_summary":    track_summary,
    "per_frame_tracks": all_tracks,
}

local_out = OUTPUT_DIR / "tracking_results.json"
with open(local_out, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved locally: {local_out}")

# ── Upload to Azure ───────────────────────────────────────────────────────────
credential   = DefaultAzureCredential()
blob_service = BlobServiceClient(
    f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
)
blob_name = "coco128/tracking/tracking_results.json"
with open(local_out, "rb") as f:
    blob_service.get_blob_client(SILVER_CONTAINER, blob_name).upload_blob(f, overwrite=True)

print(f"Uploaded → processed-silver/{blob_name}")
print(f"\nSummary:")
print(f"  Frames processed : {len(image_paths)}")
print(f"  Unique tracks    : {len(track_summary)}")
vehicle_frames = sum(1 for f in all_tracks if f['tracks'])
print(f"  Frames with vehicles tracked: {vehicle_frames}")
