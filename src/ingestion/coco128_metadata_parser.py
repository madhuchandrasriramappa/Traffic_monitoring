"""
Parses COCO128 YOLO label files and writes a normalized Parquet metadata table
to Azure Blob Storage (processed-silver layer).

YOLO label format per line: class_id cx cy w h (normalized 0-1)
COCO128 vehicle class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
"""

import os
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT_NAME"]
BRONZE_CONTAINER = "raw-bronze"
SILVER_CONTAINER = "processed-silver"
BLOB_PREFIX = "coco128/"

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 62: "tv",
    63: "laptop", 67: "cell phone", 72: "tv", 73: "laptop",
    77: "cell phone", 79: "keyboard",
}

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def get_blob_service() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    return BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )


def list_label_blobs(blob_service: BlobServiceClient) -> list[str]:
    container = blob_service.get_container_client(BRONZE_CONTAINER)
    return [
        b.name for b in container.list_blobs(name_starts_with=f"{BLOB_PREFIX}labels/")
        if b.name.endswith(".txt")
    ]


def parse_label_blob(blob_service: BlobServiceClient, blob_name: str) -> dict:
    client = blob_service.get_blob_client(container=BRONZE_CONTAINER, blob=blob_name)
    content = client.download_blob().readall().decode("utf-8").strip()

    filename = Path(blob_name).stem + ".jpg"
    rows = [line.split() for line in content.splitlines() if line.strip()]

    class_ids = [int(r[0]) for r in rows if len(r) == 5]
    vehicle_count = sum(1 for c in class_ids if c in VEHICLE_CLASS_IDS)
    categories = sorted({COCO_CLASSES.get(c, f"class_{c}") for c in class_ids})

    return {
        "filename": filename,
        "split": "train2017",
        "total_labels": len(class_ids),
        "vehicle_count": vehicle_count,
        "categories": ",".join(categories),
        "has_vehicles": vehicle_count > 0,
    }


def upload_parquet(blob_service: BlobServiceClient, df: pd.DataFrame, blob_name: str) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    client = blob_service.get_blob_client(container=SILVER_CONTAINER, blob=blob_name)
    client.upload_blob(buf, overwrite=True)
    log.info(f"Written {len(df)} rows → {SILVER_CONTAINER}/{blob_name}")


def main() -> None:
    blob_service = get_blob_service()

    log.info("Listing label blobs in raw-bronze...")
    label_blobs = list_label_blobs(blob_service)
    log.info(f"Found {len(label_blobs)} label files")

    rows = []
    for blob_name in label_blobs:
        rows.append(parse_label_blob(blob_service, blob_name))

    df = pd.DataFrame(rows)
    upload_parquet(blob_service, df, "coco128/metadata/metadata.parquet")

    log.info(f"Total images: {len(df)}")
    log.info(f"Images with vehicles: {df['has_vehicles'].sum()}")
    log.info(f"Total vehicle annotations: {df['vehicle_count'].sum()}")
    log.info(f"Category distribution:\n{df['categories'].value_counts().head(10)}")


if __name__ == "__main__":
    main()
