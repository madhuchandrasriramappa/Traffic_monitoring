"""
COCO128 → Azure Blob Storage (raw-bronze) uploader.
Uploads images and YOLO labels in parallel using chunked transfer.
"""

import os
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.identity import DefaultAzureCredential
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT_NAME"]
CONTAINER = "raw-bronze"
BLOB_PREFIX = "coco128/"
MAX_WORKERS = 8
CHUNK_SIZE = 4 * 1024 * 1024


def get_blob_service() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    return BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )


def upload_file(blob_service: BlobServiceClient, local_path: Path, blob_name: str) -> dict:
    blob_client = blob_service.get_blob_client(container=CONTAINER, blob=blob_name)

    try:
        props = blob_client.get_blob_properties()
        if props.size == local_path.stat().st_size:
            return {"file": str(local_path), "status": "skipped"}
    except Exception:
        pass

    content_type = "image/jpeg" if local_path.suffix in (".jpg", ".jpeg") else "text/plain"

    with open(local_path, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            max_concurrency=4,
            chunk_size=CHUNK_SIZE,
            content_settings=ContentSettings(content_type=content_type),
        )
    return {"file": str(local_path), "status": "uploaded", "size_bytes": local_path.stat().st_size}


def upload_directory(blob_service: BlobServiceClient, local_dir: Path, blob_prefix: str) -> list[dict]:
    files = [f for f in local_dir.rglob("*") if f.is_file()]
    log.info(f"Uploading {len(files)} files from {local_dir} → {CONTAINER}/{blob_prefix}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(upload_file, blob_service, f, blob_prefix + str(f.relative_to(local_dir))): f
            for f in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"file": str(futures[future]), "status": "error", "error": str(exc)})
    return results


def main(dataset_root: str) -> None:
    root = Path(dataset_root)
    blob_service = get_blob_service()
    all_results = []

    for subdir in ["images", "labels"]:
        local_dir = root / subdir
        if local_dir.exists():
            results = upload_directory(blob_service, local_dir, f"{BLOB_PREFIX}{subdir}/")
            all_results.extend(results)

    uploaded = sum(1 for r in all_results if r["status"] == "uploaded")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    errors = [r for r in all_results if r["status"] == "error"]

    log.info(f"Upload complete. Uploaded: {uploaded} | Skipped: {skipped} | Errors: {len(errors)}")

    manifest = {"total": len(all_results), "uploaded": uploaded, "skipped": skipped, "errors": len(errors)}
    blob_service.get_blob_client(container=CONTAINER, blob=f"{BLOB_PREFIX}upload_manifest.json").upload_blob(
        json.dumps(manifest, indent=2), overwrite=True
    )
    log.info(f"Manifest written to {CONTAINER}/{BLOB_PREFIX}upload_manifest.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python coco128_upload.py <path-to-coco128>")
        sys.exit(1)
    main(sys.argv[1])
