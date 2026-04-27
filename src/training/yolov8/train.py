"""
Sprint 04 — YOLOv8 Fine-tuning
Fine-tunes YOLOv8n on COCO128, uploads the best model weights
to Azure Blob Storage (models container) and registers it in Azure ML.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

STORAGE_ACCOUNT = "trafficproddlake"
MODELS_CONTAINER = "models"
MODEL_NAME = "yolov8-traffic"
EPOCHS = 10
IMG_SIZE = 640
DATA_YAML = Path(__file__).parent / "data.yaml"
RUN_DIR = Path(__file__).parent / "runs"


def train() -> Path:
    model = YOLO("yolov8n.pt")  # nano — smallest, fastest for CPU
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        project=str(RUN_DIR),
        name="yolov8n-traffic",
        exist_ok=True,
        device="cpu",
        workers=0,
        verbose=False,
    )
    best_weights = RUN_DIR / "yolov8n-traffic" / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")
    return best_weights


def upload_model(weights_path: Path) -> str:
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    blob_name = f"{MODEL_NAME}/v1_{timestamp}/best.pt"

    with open(weights_path, "rb") as f:
        blob_service.get_blob_client(MODELS_CONTAINER, blob_name).upload_blob(f, overwrite=True)

    print(f"Model uploaded to {MODELS_CONTAINER}/{blob_name}")
    return blob_name


def register_model_in_aml(blob_name: str, metrics: dict) -> None:
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id="61f72ce7-2f25-47f6-81ee-3772590685a1",
            resource_group_name="rg-traffic-prod",
            workspace_name="trafficprodaml",
        )

        model = Model(
            path=f"azureml://datastores/workspaceblobstore/paths/{blob_name}",
            name=MODEL_NAME,
            description="YOLOv8n fine-tuned on COCO128 for vehicle detection",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "framework": "YOLOv8",
                "dataset": "COCO128",
                "epochs": str(EPOCHS),
                "map50": str(round(metrics.get("map50", 0), 4)),
            },
        )
        registered = ml_client.models.create_or_update(model)
        print(f"Model registered in Azure ML: {registered.name} v{registered.version}")
    except Exception as e:
        print(f"AML registration skipped (will upload to blob only): {e}")


def save_training_summary(weights_path: Path, blob_name: str) -> None:
    run_dir = weights_path.parent.parent
    results_csv = run_dir / "results.csv"

    metrics = {}
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                metrics = {
                    "map50": float(last.get("metrics/mAP50(B)", 0)),
                    "map50_95": float(last.get("metrics/mAP50-95(B)", 0)),
                    "precision": float(last.get("metrics/precision(B)", 0)),
                    "recall": float(last.get("metrics/recall(B)", 0)),
                }

    summary = {
        "model": MODEL_NAME,
        "base_model": "yolov8n",
        "dataset": "COCO128",
        "epochs": EPOCHS,
        "image_size": IMG_SIZE,
        "blob_path": blob_name,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }

    summary_path = weights_path.parent.parent / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json.dumps(summary, indent=2)}")
    return metrics


def main():
    print(f"Starting YOLOv8 training — {EPOCHS} epochs on COCO128")
    weights_path = train()
    blob_name = upload_model(weights_path)
    metrics = save_training_summary(weights_path, blob_name)
    register_model_in_aml(blob_name, metrics)
    print("Sprint 04 complete.")


if __name__ == "__main__":
    main()
