"""
BDD100K → Azure Blob Storage (raw-bronze) uploader.
Uploads videos, images, and JSON metadata in parallel using chunked transfer.
"""

import os
import sys
import json
import hashlib
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
MAX_WORKERS = 8
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks

# BDD100K folder layout on local disk after extraction
BDD100K_STRUCTURE = {
    "videos": "bdd100k/videos",
    "images": "bdd100k/images/100k",
    "labels": "bdd100k/labels",
}

# Target paths in blob storage
BLOB_PREFIX = {
    "videos": "bdd100k/videos/",
    "images": "bdd100k/images/",
    "labels": "bdd100k/labels/",
}


def get_blob_client() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=credential)


def compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_file(
    blob_service: BlobServiceClient,
    local_path: Path,
    blob_name: str,
) -> dict:
    blob_client = blob_service.get_blob_client(container=CONTAINER, blob=blob_name)

    # Skip if already uploaded with matching size
    try:
        props = blob_client.get_blob_properties()
        if props.size == local_path.stat().st_size:
            return {"file": str(local_path), "status": "skipped"}
    except Exception:
        pass

    content_type = "video/mp4" if local_path.suffix == ".mp4" else "application/octet-stream"
    if local_path.suffix == ".json":
        content_type = "application/json"
    elif local_path.suffix in (".jpg", ".jpeg"):
        content_type = "image/jpeg"
    elif local_path.suffix == ".png":
        content_type = "image/png"

    with open(local_path, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            max_concurrency=4,
            chunk_size=CHUNK_SIZE,
            content_settings=ContentSettings(content_type=content_type),
        )

    return {"file": str(local_path), "status": "uploaded", "size_bytes": local_path.stat().st_size}


def upload_directory(
    blob_service: BlobServiceClient,
    local_dir: Path,
    blob_prefix: str,
    extensions: list[str] | None = None,
) -> list[dict]:
    files = [
        f for f in local_dir.rglob("*")
        if f.is_file() and (extensions is None or f.suffix in extensions)
    ]
    log.info(f"Uploading {len(files)} files from {local_dir} → {CONTAINER}/{blob_prefix}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                upload_file,
                blob_service,
                f,
                blob_prefix + str(f.relative_to(local_dir)),
            ): f
            for f in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"file": str(futures[future]), "status": "error", "error": str(exc)})
    return results


def main(bdd100k_root: str) -> None:
    root = Path(bdd100k_root)
    if not root.exists():
        log.error(f"BDD100K root not found: {root}")
        sys.exit(1)

    blob_service = get_blob_client()
    all_results = []

    for category, rel_path in BDD100K_STRUCTURE.items():
        local_dir = root / rel_path
        if not local_dir.exists():
            log.warning(f"Directory not found, skipping: {local_dir}")
            continue

        ext_map = {
            "videos": [".mp4", ".mov"],
            "images": [".jpg", ".jpeg", ".png"],
            "labels": [".json"],
        }

        results = upload_directory(
            blob_service, local_dir, BLOB_PREFIX[category], ext_map[category]
        )
        all_results.extend(results)

    uploaded = sum(1 for r in all_results if r["status"] == "uploaded")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    errors = [r for r in all_results if r["status"] == "error"]

    log.info(f"Upload complete. Uploaded: {uploaded} | Skipped: {skipped} | Errors: {len(errors)}")

    if errors:
        log.error("Failed files:")
        for e in errors:
            log.error(f"  {e['file']}: {e['error']}")
        sys.exit(1)

    # Write manifest to blob
    manifest = {
        "total_files": len(all_results),
        "uploaded": uploaded,
        "skipped": skipped,
        "errors": len(errors),
    }
    manifest_client = blob_service.get_blob_client(
        container=CONTAINER, blob="bdd100k/upload_manifest.json"
    )
    manifest_client.upload_blob(json.dumps(manifest, indent=2), overwrite=True)
    log.info("Manifest written to raw-bronze/bdd100k/upload_manifest.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bdd100k_upload.py <path-to-bdd100k-root>")
        sys.exit(1)
    main(sys.argv[1])
