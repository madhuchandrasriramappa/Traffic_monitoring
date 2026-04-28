"""
Sprint 08 — AKS Real-Time Inference Service
Consumes base64-encoded JPEG frames from Azure Event Hubs,
runs YOLOv8 vehicle detection + ConvLSTM anomaly detection,
and logs structured results per frame.
"""

import os, json, base64, time, logging
import numpy as np
import cv2
import torch
from io import BytesIO
from pathlib import Path
from collections import deque

from azure.eventhub import EventHubConsumerClient
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("inference")

# ── Config (injected via env vars in Kubernetes) ──────────────────────────────
STORAGE_ACCOUNT   = os.environ.get("STORAGE_ACCOUNT",  "trafficproddlake")
MODELS_CONTAINER  = os.environ.get("MODELS_CONTAINER", "models")
EVENTHUB_CONN_STR = os.environ.get("EVENTHUB_CONN_STR", "")
EVENTHUB_NAME     = os.environ.get("EVENTHUB_NAME",    "traffic-stream")
CONSUMER_GROUP    = os.environ.get("CONSUMER_GROUP",   "$Default")

YOLO_BLOB         = "yolov8-traffic/v1_20260427_234116/best.pt"
CONVLSTM_BLOB     = "convlstm-accident-detector/v1_20260428_012834/convlstm_best.pt"
LOCAL_MODEL_DIR   = Path("/tmp/models")

SEQ_LEN           = 4
IMG_SIZE          = 64
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
CONF_THRESHOLD    = 0.3

COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ── Model download ────────────────────────────────────────────────────────────
def download_models() -> tuple[Path, Path]:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    yolo_path     = LOCAL_MODEL_DIR / "best.pt"
    convlstm_path = LOCAL_MODEL_DIR / "convlstm_best.pt"

    credential   = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )

    for blob_name, local_path in [(YOLO_BLOB, yolo_path), (CONVLSTM_BLOB, convlstm_path)]:
        if local_path.exists():
            log.info("Already cached: %s", local_path.name)
            continue
        log.info("Downloading %s ...", blob_name)
        data = blob_service.get_blob_client(MODELS_CONTAINER, blob_name).download_blob().readall()
        local_path.write_bytes(data)
        log.info("Saved → %s", local_path)

    return yolo_path, convlstm_path


# ── ConvLSTM loader (reuse model.py definition inline to avoid path issues) ──
class ConvLSTMCell(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = torch.nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMAnomalyDetector(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, img_size=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.convlstm = ConvLSTMCell(32, hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H // 4, W // 4)
        c = torch.zeros(B, self.hidden_channels, H // 4, W // 4)
        for t in range(T):
            enc = self.encoder(x[:, t])
            h, c = self.convlstm(enc, h, c)
        return self.classifier(h)


# ── Inference engine ──────────────────────────────────────────────────────────
class InferenceEngine:
    def __init__(self, yolo_path: Path, convlstm_path: Path):
        from ultralytics import YOLO
        self.yolo = YOLO(str(yolo_path))
        log.info("YOLOv8 loaded")

        self.convlstm = ConvLSTMAnomalyDetector()
        self.convlstm.load_state_dict(torch.load(str(convlstm_path), map_location="cpu"))
        self.convlstm.eval()
        log.info("ConvLSTM loaded")

        self.frame_buffer: deque = deque(maxlen=SEQ_LEN)
        self.frame_count = 0

    def process(self, frame_bgr: np.ndarray) -> dict:
        self.frame_count += 1

        # YOLOv8 detection
        results   = self.yolo(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": COCO_NAMES.get(cls_id, "vehicle"),
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [x1, y1, x2, y2],
            })

        # Buffer frame for ConvLSTM
        resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        tensor  = torch.from_numpy(resized.transpose(2, 0, 1))
        self.frame_buffer.append(tensor)

        anomaly_score = None
        anomaly_flag  = False
        if len(self.frame_buffer) == SEQ_LEN:
            seq   = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0)  # (1,T,C,H,W)
            with torch.no_grad():
                logit = self.convlstm(seq)
                score = torch.sigmoid(logit).item()
            anomaly_score = round(score, 4)
            anomaly_flag  = score > 0.5

        return {
            "frame":         self.frame_count,
            "vehicles":      len(detections),
            "detections":    detections,
            "anomaly_score": anomaly_score,
            "accident":      anomaly_flag,
        }


# ── Event Hub consumer ────────────────────────────────────────────────────────
def on_event(partition_context, event, engine: InferenceEngine):
    try:
        body = event.body_as_str()
        msg  = json.loads(body)

        # Decode base64 JPEG or accept raw npy blob reference
        if "image_b64" in msg:
            img_bytes = base64.b64decode(msg["image_b64"])
            arr       = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            log.warning("Event missing 'image_b64' field — skipping")
            partition_context.update_checkpoint(event)
            return

        result = engine.process(frame_bgr)
        log.info(json.dumps(result))

        if result["accident"]:
            log.warning("ACCIDENT DETECTED — frame %d | score=%.4f",
                        result["frame"], result["anomaly_score"])

    except Exception as exc:
        log.error("Error processing event: %s", exc)

    partition_context.update_checkpoint(event)


def main():
    log.info("Sprint 08 — AKS Inference Service starting")

    yolo_path, convlstm_path = download_models()
    engine = InferenceEngine(yolo_path, convlstm_path)

    if not EVENTHUB_CONN_STR:
        log.warning("EVENTHUB_CONN_STR not set — running in offline demo mode")
        _offline_demo(engine)
        return

    log.info("Connecting to Event Hub '%s'...", EVENTHUB_NAME)
    client = EventHubConsumerClient.from_connection_string(
        EVENTHUB_CONN_STR,
        consumer_group=CONSUMER_GROUP,
        eventhub_name=EVENTHUB_NAME,
    )
    log.info("Listening for frames...")
    with client:
        client.receive(
            on_event=lambda pc, ev: on_event(pc, ev, engine),
            starting_position="-1",
        )


def _offline_demo(engine: InferenceEngine):
    """Process local COCO128 images when no Event Hub is configured."""
    images_dir = Path(__file__).parents[2] / "data/coco128/images/train2017"
    paths = sorted(images_dir.glob("*.jpg"))[:20]
    if not paths:
        log.error("No local images found at %s", images_dir)
        return

    log.info("Offline demo: processing %d local frames", len(paths))
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        result = engine.process(frame)
        log.info(json.dumps(result))
        time.sleep(0.1)

    log.info("Offline demo complete.")


if __name__ == "__main__":
    main()
