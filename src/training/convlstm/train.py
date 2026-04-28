"""
Sprint 06 — ConvLSTM Accident Detection Training
Builds frame sequences from processed COCO128 .npy files,
labels them using vehicle density heuristic, trains ConvLSTM,
then uploads the model to Azure ML.
"""

import os, json, numpy as np, torch
from pathlib import Path
from datetime import datetime
from io import BytesIO
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from model import ConvLSTMAnomalyDetector
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# ── Config ────────────────────────────────────────────────────────────────────
STORAGE_ACCOUNT  = "trafficproddlake"
SILVER_CONTAINER = "processed-silver"
MODELS_CONTAINER = "models"
PROCESSED_DIR    = Path(__file__).parents[4] / "data" / "processed_frames"
SEQ_LEN          = 4       # frames per sequence
IMG_SIZE         = 64      # resize to 64x64 for speed
EPOCHS           = 15
BATCH_SIZE       = 4
LR               = 1e-3
DEVICE           = "cpu"

# ── Load processed frames locally ─────────────────────────────────────────────
# Download .npy frames from processed-silver if not cached locally
def get_frames() -> list[np.ndarray]:
    cache_dir = Path(__file__).parent / "frame_cache"
    cache_dir.mkdir(exist_ok=True)

    cached = sorted(cache_dir.glob("*.npy"))
    if len(cached) >= 128:
        print(f"Loading {len(cached)} cached frames...")
        return [np.load(f) for f in cached]

    print("Downloading processed frames from Azure...")
    credential   = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )
    container = blob_service.get_container_client(SILVER_CONTAINER)
    blobs = [b.name for b in container.list_blobs(name_starts_with="coco128/processed/") if b.name.endswith(".npy")]

    frames = []
    for blob_name in sorted(blobs):
        data = blob_service.get_blob_client(SILVER_CONTAINER, blob_name).download_blob().readall()
        arr  = np.load(BytesIO(data))   # (640, 640, 3) float32
        fname = cache_dir / Path(blob_name).name
        np.save(fname, arr)
        frames.append(arr)

    print(f"Downloaded {len(frames)} frames.")
    return frames


# ── Dataset ───────────────────────────────────────────────────────────────────
import cv2

class FrameSequenceDataset(Dataset):
    def __init__(self, frames: list[np.ndarray], seq_len: int, img_size: int):
        self.seq_len  = seq_len
        self.img_size = img_size

        # Build overlapping sequences
        self.sequences = []
        self.labels    = []

        for i in range(len(frames) - seq_len + 1):
            seq = frames[i : i + seq_len]
            # Anomaly heuristic: high pixel variance in last frame vs first → sudden change
            motion = float(np.mean(np.abs(seq[-1] - seq[0])))
            label  = 1.0 if motion > 0.15 else 0.0
            self.sequences.append(seq)
            self.labels.append(label)

        pos = sum(self.labels)
        print(f"Dataset: {len(self.sequences)} sequences | anomalies: {int(pos)} | normal: {int(len(self.labels)-pos)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = self.sequences[idx]
        label = self.labels[idx]

        frames_t = []
        for frame in seq:
            # Resize to img_size x img_size
            resized = cv2.resize(frame, (self.img_size, self.img_size))
            # (H,W,C) BGR float32 → (C,H,W)
            tensor  = torch.from_numpy(resized.transpose(2, 0, 1))
            frames_t.append(tensor)

        x = torch.stack(frames_t, dim=0)          # (T, C, H, W)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(dataset: FrameSequenceDataset) -> tuple[ConvLSTMAnomalyDetector, dict]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model  = ConvLSTMAnomalyDetector(in_channels=3, hidden_channels=32, img_size=IMG_SIZE)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds   = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += len(y)

        scheduler.step()
        avg_loss = total_loss / total
        acc      = correct / total
        history.append({"epoch": epoch, "loss": round(avg_loss, 4), "accuracy": round(acc, 4)})
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.4f}")

    return model, history


# ── Upload & Register ─────────────────────────────────────────────────────────
def save_and_upload(model: ConvLSTMAnomalyDetector, history: list[dict]) -> str:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    weights_path = out_dir / "convlstm_best.pt"
    torch.save(model.state_dict(), weights_path)

    credential   = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    blob_name = f"convlstm-accident-detector/v1_{timestamp}/convlstm_best.pt"

    with open(weights_path, "rb") as f:
        blob_service.get_blob_client(MODELS_CONTAINER, blob_name).upload_blob(f, overwrite=True)
    print(f"Model uploaded → models/{blob_name}")
    return blob_name


def register_in_aml(blob_name: str, history: list[dict]) -> None:
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id="61f72ce7-2f25-47f6-81ee-3772590685a1",
            resource_group_name="rg-traffic-prod",
            workspace_name="trafficprodaml",
        )
        last = history[-1]
        model = Model(
            path=str(Path(__file__).parent / "output" / "convlstm_best.pt"),
            name="convlstm-accident-detector",
            description="ConvLSTM spatio-temporal anomaly detector for traffic accidents",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "framework":  "PyTorch",
                "dataset":    "COCO128-sequences",
                "epochs":     str(EPOCHS),
                "final_acc":  str(last["accuracy"]),
                "final_loss": str(last["loss"]),
            },
        )
        registered = ml_client.models.create_or_update(model)
        print(f"Registered in Azure ML: {registered.name} v{registered.version}")
    except Exception as e:
        print(f"AML registration error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Sprint 06 — ConvLSTM Accident Detection")
    frames  = get_frames()
    dataset = FrameSequenceDataset(frames, SEQ_LEN, IMG_SIZE)

    print(f"\nTraining ConvLSTM for {EPOCHS} epochs...")
    model, history = train_model(dataset)

    blob_name = save_and_upload(model, history)
    register_in_aml(blob_name, history)

    summary = {
        "model":         "convlstm-accident-detector",
        "seq_len":       SEQ_LEN,
        "img_size":      IMG_SIZE,
        "epochs":        EPOCHS,
        "final_loss":    history[-1]["loss"],
        "final_accuracy":history[-1]["accuracy"],
        "blob_path":     blob_name,
        "trained_at":    datetime.utcnow().isoformat(),
    }
    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
    print("Sprint 06 complete.")


if __name__ == "__main__":
    main()
