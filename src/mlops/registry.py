"""
Sprint 07 — MLOps Model Registry
Logs YOLOv8 and ConvLSTM training runs as MLflow experiments in Azure ML,
registers both models in the Azure ML Model Registry, and promotes the
latest version of each to Production.
"""

import mlflow
import json
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# ── Config ────────────────────────────────────────────────────────────────────
SUBSCRIPTION_ID    = "61f72ce7-2f25-47f6-81ee-3772590685a1"
RESOURCE_GROUP     = "rg-traffic-prod"
WORKSPACE_NAME     = "trafficprodaml"

YOLO_WEIGHTS       = Path(__file__).parents[2] / "src/training/yolov8/runs/yolov8n-traffic/weights/best.pt"
CONVLSTM_WEIGHTS   = Path(__file__).parents[2] / "src/training/convlstm/output/convlstm_best.pt"

# Metrics captured from Sprint 04 and Sprint 06
YOLO_METRICS = {
    "mAP50":     0.698,
    "precision": 0.693,
    "recall":    0.636,
    "epochs":    10,
}
CONVLSTM_METRICS = {
    "final_loss":     0.0014,
    "final_accuracy": 1.0,
    "epochs":         15,
    "seq_len":        4,
    "img_size":       64,
}


# ── Azure ML client ───────────────────────────────────────────────────────────
def get_ml_client() -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


# ── MLflow experiment logging ─────────────────────────────────────────────────
def log_yolov8_experiment(ml_client: MLClient) -> str:
    tracking_uri = ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("yolov8-traffic-detection")

    with mlflow.start_run(run_name="yolov8n-coco128-v1") as run:
        mlflow.log_params({
            "model":     "yolov8n",
            "dataset":   "COCO128",
            "epochs":    YOLO_METRICS["epochs"],
            "optimizer": "SGD",
            "device":    "cpu",
        })
        mlflow.log_metrics({
            "mAP50":     YOLO_METRICS["mAP50"],
            "precision": YOLO_METRICS["precision"],
            "recall":    YOLO_METRICS["recall"],
        })
        if YOLO_WEIGHTS.exists():
            mlflow.log_artifact(str(YOLO_WEIGHTS), artifact_path="weights")

        run_id = run.info.run_id
        print(f"YOLOv8 run logged  → run_id: {run_id}")
        return run_id


def log_convlstm_experiment(ml_client: MLClient) -> str:
    tracking_uri = ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("convlstm-accident-detection")

    with mlflow.start_run(run_name="convlstm-coco128-v1") as run:
        mlflow.log_params({
            "model":          "ConvLSTMAnomalyDetector",
            "dataset":        "COCO128-sequences",
            "epochs":         CONVLSTM_METRICS["epochs"],
            "seq_len":        CONVLSTM_METRICS["seq_len"],
            "img_size":       CONVLSTM_METRICS["img_size"],
            "hidden_channels": 32,
            "optimizer":      "Adam",
            "lr":             0.001,
            "loss":           "BCEWithLogitsLoss",
            "device":         "cpu",
        })
        mlflow.log_metrics({
            "final_loss":     CONVLSTM_METRICS["final_loss"],
            "final_accuracy": CONVLSTM_METRICS["final_accuracy"],
        })
        if CONVLSTM_WEIGHTS.exists():
            mlflow.log_artifact(str(CONVLSTM_WEIGHTS), artifact_path="weights")

        run_id = run.info.run_id
        print(f"ConvLSTM run logged → run_id: {run_id}")
        return run_id


# ── Register & promote models ─────────────────────────────────────────────────
def register_and_promote(ml_client: MLClient) -> None:
    models_to_register = [
        {
            "path":        str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else None,
            "name":        "yolov8-traffic",
            "description": "YOLOv8n vehicle detector fine-tuned on COCO128",
            "tags": {
                "framework":   "Ultralytics",
                "task":        "object-detection",
                "dataset":     "COCO128",
                "mAP50":       str(YOLO_METRICS["mAP50"]),
                "stage":       "Production",
            },
        },
        {
            "path":        str(CONVLSTM_WEIGHTS) if CONVLSTM_WEIGHTS.exists() else None,
            "name":        "convlstm-accident-detector",
            "description": "ConvLSTM spatio-temporal anomaly detector for traffic accidents",
            "tags": {
                "framework":  "PyTorch",
                "task":       "anomaly-detection",
                "dataset":    "COCO128-sequences",
                "final_loss": str(CONVLSTM_METRICS["final_loss"]),
                "stage":      "Production",
            },
        },
    ]

    for spec in models_to_register:
        if spec["path"] is None:
            print(f"  Skipping {spec['name']} — weights file not found locally")
            continue

        model = Model(
            path=spec["path"],
            name=spec["name"],
            description=spec["description"],
            type=AssetTypes.CUSTOM_MODEL,
            tags=spec["tags"],
        )
        registered = ml_client.models.create_or_update(model)
        print(f"  Registered: {registered.name}  version={registered.version}  stage=Production")


# ── Summary report ────────────────────────────────────────────────────────────
def print_registry_summary(ml_client: MLClient) -> None:
    print("\n── Model Registry Summary ──────────────────────────────────────────")
    for model_name in ["yolov8-traffic", "convlstm-accident-detector"]:
        versions = list(ml_client.models.list(name=model_name))
        if not versions:
            print(f"  {model_name}: (no versions found)")
            continue
        latest = max(versions, key=lambda m: int(m.version))
        print(f"  {model_name}")
        print(f"    latest version : {latest.version}")
        print(f"    stage tag      : {latest.tags.get('stage', 'N/A')}")
        print(f"    description    : {latest.description}")
    print("────────────────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Sprint 07 — MLOps Model Registry")
    ml_client = get_ml_client()

    print("\n[1/3] Logging MLflow experiments...")
    log_yolov8_experiment(ml_client)
    log_convlstm_experiment(ml_client)

    print("\n[2/3] Registering & promoting models to Production...")
    register_and_promote(ml_client)

    print("\n[3/3] Registry summary:")
    print_registry_summary(ml_client)

    print("\nSprint 07 complete.")


if __name__ == "__main__":
    main()
