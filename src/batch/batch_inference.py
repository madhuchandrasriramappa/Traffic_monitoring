"""
Sprint 09 — Batch Processing Pipeline
Reads all processed .npy frames from processed-silver, runs YOLOv8 vehicle
detection + ConvLSTM anomaly detection on every 4-frame window, and writes
structured results to gold-serving for downstream reporting.
"""

import os, json, sys, time
import numpy as np
import cv2
import torch
from io import BytesIO
from pathlib import Path
from collections import deque
from datetime import datetime

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.identity import DefaultAzureCredential

# ── Config ────────────────────────────────────────────────────────────────────
STORAGE_ACCOUNT  = "trafficproddlake"
SILVER_CONTAINER = "processed-silver"
MODELS_CONTAINER = "models"
GOLD_CONTAINER   = "gold-serving"

YOLO_BLOB      = "yolov8-traffic/v1_20260427_234116/best.pt"
CONVLSTM_BLOB  = "convlstm-accident-detector/v1_20260428_012834/convlstm_best.pt"
LOCAL_CACHE    = Path(__file__).parent / "model_cache"

SEQ_LEN           = 4
IMG_SIZE          = 64
CONF_THRESHOLD    = 0.3
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
COCO_NAMES        = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ── Model definitions (self-contained, no external model.py dependency) ───────
class ConvLSTMCell(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = torch.nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad
        )

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMAnomalyDetector(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, img_size=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),           torch.nn.ReLU(),
        )
        self.convlstm = ConvLSTMCell(32, hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(hidden_channels, 64), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H // 4, W // 4)
        c = torch.zeros(B, self.hidden_channels, H // 4, W // 4)
        for t in range(T):
            enc = self.encoder(x[:, t])
            h, c = self.convlstm(enc, h, c)
        return self.classifier(h)


# ── Azure helpers ─────────────────────────────────────────────────────────────
def get_blob_service() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    return BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )


def download_models(blob_service: BlobServiceClient) -> tuple[Path, Path]:
    LOCAL_CACHE.mkdir(exist_ok=True)
    yolo_path     = LOCAL_CACHE / "best.pt"
    convlstm_path = LOCAL_CACHE / "convlstm_best.pt"

    for blob_name, local_path in [(YOLO_BLOB, yolo_path), (CONVLSTM_BLOB, convlstm_path)]:
        if local_path.exists():
            print(f"  Cached: {local_path.name}")
            continue
        print(f"  Downloading {blob_name} ...")
        data = blob_service.get_blob_client(MODELS_CONTAINER, blob_name).download_blob().readall()
        local_path.write_bytes(data)
    return yolo_path, convlstm_path


def load_silver_frames(blob_service: BlobServiceClient) -> list[tuple[str, np.ndarray]]:
    container = blob_service.get_container_client(SILVER_CONTAINER)
    blobs = sorted(
        b.name for b in container.list_blobs(name_starts_with="coco128/processed/")
        if b.name.endswith(".npy")
    )
    frames = []
    for blob_name in blobs:
        data = blob_service.get_blob_client(SILVER_CONTAINER, blob_name).download_blob().readall()
        arr  = np.load(BytesIO(data))   # (640, 640, 3) float32 0–1
        bgr  = (arr * 255).astype(np.uint8)
        frames.append((Path(blob_name).stem, bgr))
    print(f"  Loaded {len(frames)} frames from processed-silver")
    return frames


# ── Inference helpers ─────────────────────────────────────────────────────────
def run_yolo(yolo_model, frame_bgr: np.ndarray) -> list[dict]:
    results = yolo_model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "class":      COCO_NAMES.get(cls_id, "vehicle"),
            "confidence": round(float(box.conf[0]), 3),
            "bbox":       [x1, y1, x2, y2],
        })
    return detections


def run_convlstm(convlstm_model, buffer: deque) -> float | None:
    if len(buffer) < SEQ_LEN:
        return None
    seq = torch.stack(list(buffer), dim=0).unsqueeze(0)   # (1, T, C, H, W)
    with torch.no_grad():
        logit = convlstm_model(seq)
    return round(torch.sigmoid(logit).item(), 4)


# ── Main batch loop ───────────────────────────────────────────────────────────
def run_batch(blob_service: BlobServiceClient) -> dict:
    from ultralytics import YOLO

    print("\n[1/4] Downloading models...")
    yolo_path, convlstm_path = download_models(blob_service)

    print("[2/4] Loading models...")
    yolo_model     = YOLO(str(yolo_path))
    convlstm_model = ConvLSTMAnomalyDetector()
    convlstm_model.load_state_dict(torch.load(str(convlstm_path), map_location="cpu"))
    convlstm_model.eval()

    print("[3/4] Loading frames from Silver layer...")
    frames = load_silver_frames(blob_service)

    print(f"[4/4] Running inference on {len(frames)} frames...")
    frame_buffer: deque = deque(maxlen=SEQ_LEN)
    frame_results = []
    total_vehicles = 0
    total_accidents = 0

    for idx, (frame_name, frame_bgr) in enumerate(frames):
        detections = run_yolo(yolo_model, frame_bgr)

        resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        frame_buffer.append(torch.from_numpy(resized.transpose(2, 0, 1)))
        anomaly_score = run_convlstm(convlstm_model, frame_buffer)
        accident      = bool(anomaly_score is not None and anomaly_score > 0.5)

        total_vehicles  += len(detections)
        total_accidents += int(accident)

        frame_results.append({
            "frame_index":   idx,
            "frame_name":    frame_name,
            "vehicles":      len(detections),
            "detections":    detections,
            "anomaly_score": anomaly_score,
            "accident":      accident,
        })

        if (idx + 1) % 32 == 0 or idx == len(frames) - 1:
            print(f"  Processed {idx + 1}/{len(frames)} frames | "
                  f"vehicles={total_vehicles} | accidents={total_accidents}")

    return {
        "run_timestamp":    datetime.utcnow().isoformat(),
        "dataset":          "COCO128",
        "total_frames":     len(frames),
        "total_vehicles":   total_vehicles,
        "total_accidents":  total_accidents,
        "accident_rate":    round(total_accidents / max(len(frames), 1), 4),
        "frames":           frame_results,
    }


# ── Write to Gold layer ───────────────────────────────────────────────────────
def write_gold(blob_service: BlobServiceClient, results: dict) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Full results
    full_blob  = f"coco128/batch_results/{timestamp}/full_results.json"
    full_bytes = json.dumps(results, indent=2).encode()
    blob_service.get_blob_client(GOLD_CONTAINER, full_blob).upload_blob(full_bytes, overwrite=True)
    print(f"  Full results → gold-serving/{full_blob}")

    # Summary (no per-frame detail)
    summary = {k: v for k, v in results.items() if k != "frames"}
    summary_blob  = f"coco128/batch_results/{timestamp}/summary.json"
    summary_bytes = json.dumps(summary, indent=2).encode()
    blob_service.get_blob_client(GOLD_CONTAINER, summary_blob).upload_blob(summary_bytes, overwrite=True)
    print(f"  Summary      → gold-serving/{summary_blob}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("Sprint 09 — Batch Processing Pipeline")
    t0 = time.time()

    blob_service = get_blob_service()
    results      = run_batch(blob_service)

    print("\nWriting results to Gold layer...")
    write_gold(blob_service, results)

    elapsed = round(time.time() - t0, 1)
    print(f"\nBatch complete in {elapsed}s")
    print(json.dumps({k: v for k, v in results.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
