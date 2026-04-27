"""
Parses BDD100K JSON label files and writes a normalized Parquet metadata table
to Azure Blob Storage (processed-silver layer).

BDD100K label schema:
  name         → video/image filename
  attributes   → weather, scene, timeofday
  labels[]     → category, box2d, id
"""

import os
import json
import logging
from pathlib import Path
from io import BytesIO

import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT_NAME"]
BRONZE_CONTAINER = "raw-bronze"
SILVER_CONTAINER = "processed-silver"

LABEL_BLOBS = [
    "bdd100k/labels/bdd100k_labels_images_train.json",
    "bdd100k/labels/bdd100k_labels_images_val.json",
]

VEHICLE_CATEGORIES = {"car", "truck", "bus", "motorcycle", "bicycle", "trailer"}


def get_blob_service() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    return BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )


def download_json(blob_service: BlobServiceClient, blob_name: str) -> list[dict]:
    client = blob_service.get_blob_client(container=BRONZE_CONTAINER, blob=blob_name)
    data = client.download_blob().readall()
    return json.loads(data)


def parse_label_file(records: list[dict], split: str) -> pd.DataFrame:
    rows = []
    for record in records:
        filename = record.get("name", "")
        attrs = record.get("attributes", {})
        weather = attrs.get("weather", "unknown")
        scene = attrs.get("scene", "unknown")
        timeofday = attrs.get("timeofday", "unknown")

        labels = record.get("labels", []) or []
        vehicle_count = sum(1 for lbl in labels if lbl.get("category") in VEHICLE_CATEGORIES)
        categories_present = list({lbl.get("category") for lbl in labels if lbl.get("category")})

        rows.append({
            "filename": filename,
            "split": split,
            "weather": weather,
            "scene": scene,
            "timeofday": timeofday,
            "total_labels": len(labels),
            "vehicle_count": vehicle_count,
            "categories": ",".join(sorted(categories_present)),
        })

    return pd.DataFrame(rows)


def upload_parquet(blob_service: BlobServiceClient, df: pd.DataFrame, blob_name: str) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    client = blob_service.get_blob_client(container=SILVER_CONTAINER, blob=blob_name)
    client.upload_blob(buf, overwrite=True)
    log.info(f"Written {len(df)} rows → {SILVER_CONTAINER}/{blob_name}")


def main() -> None:
    blob_service = get_blob_service()
    all_frames = []

    split_map = {
        "bdd100k/labels/bdd100k_labels_images_train.json": "train",
        "bdd100k/labels/bdd100k_labels_images_val.json": "val",
    }

    for blob_name in LABEL_BLOBS:
        log.info(f"Downloading {blob_name}...")
        records = download_json(blob_service, blob_name)
        split = split_map[blob_name]
        df = parse_label_file(records, split)
        all_frames.append(df)
        upload_parquet(blob_service, df, f"bdd100k/metadata/{split}_metadata.parquet")

    combined = pd.concat(all_frames, ignore_index=True)
    upload_parquet(blob_service, combined, "bdd100k/metadata/all_metadata.parquet")

    log.info("Metadata parsing complete.")
    log.info(f"Total records: {len(combined)}")
    log.info(f"Weather distribution:\n{combined['weather'].value_counts()}")
    log.info(f"Scene distribution:\n{combined['scene'].value_counts()}")


if __name__ == "__main__":
    main()
