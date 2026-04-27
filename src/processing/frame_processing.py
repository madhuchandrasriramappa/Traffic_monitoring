# Databricks notebook source
# Sprint 3 — Frame Processing
# Reads COCO128 images from raw-bronze, resizes to 640x640,
# normalizes pixel values, writes to processed-silver.

# COMMAND ----------
# Install dependencies
import subprocess
subprocess.run(["pip", "install", "opencv-python-headless", "azure-storage-blob", "azure-identity"], check=True, capture_output=True)

# COMMAND ----------
import os
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

STORAGE_ACCOUNT = "trafficproddlake"
BRONZE_CONTAINER = "raw-bronze"
SILVER_CONTAINER = "processed-silver"
IMAGE_SIZE = 640  # YOLOv8 standard input size

credential = DefaultAzureCredential()
blob_service = BlobServiceClient(
    f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=credential
)

print(f"Connected to storage account: {STORAGE_ACCOUNT}")

# COMMAND ----------
# List all images in raw-bronze/coco128/images/
container = blob_service.get_container_client(BRONZE_CONTAINER)
image_blobs = [
    b.name for b in container.list_blobs(name_starts_with="coco128/images/")
    if b.name.endswith(".jpg")
]
print(f"Found {len(image_blobs)} images to process")

# COMMAND ----------
def process_image(blob_name: str) -> dict:
    """Download, resize to 640x640, normalize, upload to silver."""
    # Download from bronze
    src = blob_service.get_blob_client(BRONZE_CONTAINER, blob_name)
    raw = src.download_blob().readall()

    # Decode image
    img_array = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return {"blob": blob_name, "status": "error", "reason": "decode failed"}

    original_h, original_w = img.shape[:2]

    # Resize to 640x640
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # Normalize pixel values to [0, 1] and save as float32 NPY
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Save processed image as .npy in silver
    filename = Path(blob_name).stem
    silver_blob_name = f"coco128/processed/{filename}.npy"

    buf = BytesIO()
    np.save(buf, img_normalized)
    buf.seek(0)

    dst = blob_service.get_blob_client(SILVER_CONTAINER, silver_blob_name)
    dst.upload_blob(buf, overwrite=True)

    return {
        "blob": blob_name,
        "status": "processed",
        "original_w": original_w,
        "original_h": original_h,
        "output": silver_blob_name
    }

# COMMAND ----------
# Process all images
results = []
errors = []

for i, blob_name in enumerate(image_blobs):
    result = process_image(blob_name)
    results.append(result)
    if result["status"] == "error":
        errors.append(result)
    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(image_blobs)} images...")

print(f"\nDone. Processed: {len(results) - len(errors)} | Errors: {len(errors)}")

# COMMAND ----------
# Write summary metadata to silver
import json
from datetime import datetime

summary = {
    "run_timestamp": datetime.utcnow().isoformat(),
    "total_images": len(image_blobs),
    "processed": len(results) - len(errors),
    "errors": len(errors),
    "output_size": IMAGE_SIZE,
    "normalization": "pixel / 255.0",
    "format": "numpy float32 (H, W, C) BGR"
}

summary_blob = blob_service.get_blob_client(SILVER_CONTAINER, "coco128/processed/processing_summary.json")
summary_blob.upload_blob(json.dumps(summary, indent=2), overwrite=True)

print("Summary written to processed-silver/coco128/processed/processing_summary.json")
print(json.dumps(summary, indent=2))
