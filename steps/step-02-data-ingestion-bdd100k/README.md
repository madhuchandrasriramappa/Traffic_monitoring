# Step 02 — BDD100K Data Ingestion

## Objective

Download the BDD100K dataset and land it in Azure Blob Storage (raw-bronze layer) in its original form. Parse and normalize the JSON annotation files into a structured Parquet metadata table stored in the processed-silver layer. Register an Azure Data Factory pipeline that orchestrates and schedules this ingestion end-to-end. This is the data foundation — every training, processing, and inference step downstream depends on this being correct and complete.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    BDD100K Dataset (local / VM)                  │
│                                                                  │
│   videos/         images/100k/       labels/                     │
│   train/ val/     train/ val/ test/  *.json                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │  bdd100k_upload.py (parallel upload)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│         Azure Blob Storage — raw-bronze container                │
│                                                                  │
│   bdd100k/videos/train/        ← .mp4 files                      │
│   bdd100k/videos/val/                                            │
│   bdd100k/images/train/        ← .jpg files                      │
│   bdd100k/images/val/                                            │
│   bdd100k/images/test/                                           │
│   bdd100k/labels/              ← original JSON annotations       │
│   bdd100k/upload_manifest.json ← upload audit record             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           │  Azure Data Factory triggers
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Azure Databricks                              │
│              metadata_parser notebook                            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│         Azure Blob Storage — processed-silver container          │
│                                                                  │
│   bdd100k/metadata/train_metadata.parquet                        │
│   bdd100k/metadata/val_metadata.parquet                          │
│   bdd100k/metadata/all_metadata.parquet                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## BDD100K Dataset Overview

| Subset | Content | Size |
|--------|---------|------|
| Videos | 100K driving videos, 40-second clips at 30fps | ~1.8 TB |
| Images | 100K keyframes (1280×720) | ~6.5 GB |
| Labels | Object detection + lane/drivable annotations (JSON) | ~300 MB |
| Metadata | Per-image: weather, scene, time-of-day | embedded in JSON |

**Download source:** [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) — requires free account registration.

---

## Setup Instructions

### 1. Install Python dependencies

```bash
cd src/ingestion
pip install -r requirements.txt
```

### 2. Authenticate to Azure

```bash
az login
# Set the correct subscription
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
```

### 3. Set environment variable

```bash
export STORAGE_ACCOUNT_NAME="trafficprodlake"
```

### 4. Download BDD100K

After registering at bdd-data.berkeley.edu, download and extract:

```
bdd100k/
├── videos/
│   ├── train/     ← 70K .mp4 files
│   └── val/       ← 10K .mp4 files
├── images/
│   └── 100k/
│       ├── train/ ← 70K .jpg files
│       ├── val/   ← 10K .jpg files
│       └── test/  ← 20K .jpg files
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

---

## Implementation Steps

### Step A — Upload Raw Data to Bronze Layer

```bash
# Upload all videos, images, and labels
python src/ingestion/bdd100k_upload.py /path/to/bdd100k
```

This script:
- Authenticates via `DefaultAzureCredential` (uses your `az login` session)
- Skips files already uploaded with matching sizes (idempotent)
- Uses 8 parallel threads with 4 MB chunked upload
- Writes `upload_manifest.json` to blob on completion

**Expected runtime:** 3–6 hours for full dataset (depends on network speed). Run on an Azure VM in the same region as your storage account to avoid egress costs.

> Recommended: Spin up a temporary `Standard_D4s_v3` VM in `eastus2`, download BDD100K directly to the VM, then run the upload script from there.

```bash
# Create a temporary ingestion VM (run from your local terminal)
az vm create \
  --resource-group rg-traffic-prod \
  --name vm-ingestion \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --location eastus2 \
  --admin-username azureuser \
  --generate-ssh-keys

# SSH into the VM and run the upload there
ssh azureuser@<VM_PUBLIC_IP>
```

### Step B — Parse Metadata to Silver Layer

```bash
python src/ingestion/metadata_parser.py
```

This script reads the two BDD100K JSON label files from `raw-bronze`, parses every annotation record, and writes three Parquet files to `processed-silver`:

| Output File | Rows | Description |
|-------------|------|-------------|
| `train_metadata.parquet` | 70,000 | Training split metadata |
| `val_metadata.parquet` | 10,000 | Validation split metadata |
| `all_metadata.parquet` | 80,000 | Combined for analysis |

**Parquet schema:**

| Column | Type | Example |
|--------|------|---------|
| `filename` | string | `0000f77c-6257be58.jpg` |
| `split` | string | `train` |
| `weather` | string | `clear`, `rainy`, `foggy` |
| `scene` | string | `highway`, `city street` |
| `timeofday` | string | `daytime`, `night` |
| `total_labels` | int | 12 |
| `vehicle_count` | int | 7 |
| `categories` | string | `car,truck,person` |

### Step C — Register ADF Pipeline

```bash
# Import the pipeline definition into Azure Data Factory
az datafactory pipeline create \
  --factory-name trafficprodadf \
  --resource-group rg-traffic-prod \
  --name pipeline_bdd100k_ingestion \
  --pipeline @deploy/adf/pipeline_bdd100k_ingestion.json
```

### Step D — Create ADF Linked Services (Databricks + Storage)

In the Azure Portal → Data Factory → Manage → Linked Services:

1. **ls_databricks** — type: Azure Databricks, point to `trafficprodadb` workspace, use existing cluster token
2. **ls_azure_blob_bronze** — type: Azure Blob Storage, point to `raw-bronze` container
3. **ls_azure_blob_silver** — type: Azure Blob Storage, point to `processed-silver` container

Or via CLI:

```bash
# Storage linked service
az datafactory linked-service create \
  --factory-name trafficprodadf \
  --resource-group rg-traffic-prod \
  --name ls_azure_blob_bronze \
  --properties '{
    "type": "AzureBlobStorage",
    "typeProperties": {
      "connectionString": "@Microsoft.KeyVault(SecretUri=https://trafficprodkv.vault.azure.net/secrets/storage-connection-string/)"
    }
  }'
```

---

## Validation Procedure

### 1. Verify bronze container has the correct file counts

```bash
# Count uploaded videos
az storage blob list \
  --account-name trafficprodlake \
  --container-name raw-bronze \
  --prefix "bdd100k/videos/train/" \
  --auth-mode login \
  --query "length(@)" -o tsv
# Expected: ~70000

# Count uploaded images
az storage blob list \
  --account-name trafficprodlake \
  --container-name raw-bronze \
  --prefix "bdd100k/images/train/" \
  --auth-mode login \
  --query "length(@)" -o tsv
# Expected: ~70000
```

### 2. Verify upload manifest

```bash
az storage blob download \
  --account-name trafficprodlake \
  --container-name raw-bronze \
  --name bdd100k/upload_manifest.json \
  --file /tmp/manifest.json \
  --auth-mode login
cat /tmp/manifest.json
# Expected: { "errors": 0, "uploaded": ..., "skipped": ... }
```

### 3. Verify silver metadata Parquet files

```bash
az storage blob list \
  --account-name trafficprodlake \
  --container-name processed-silver \
  --prefix "bdd100k/metadata/" \
  --auth-mode login \
  --output table
# Expected: 3 Parquet files present
```

### 4. Spot-check Parquet content (Python)

```python
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from io import BytesIO

blob_service = BlobServiceClient(
    "https://trafficprodlake.blob.core.windows.net",
    credential=DefaultAzureCredential()
)
client = blob_service.get_blob_client("processed-silver", "bdd100k/metadata/all_metadata.parquet")
df = pd.read_parquet(BytesIO(client.download_blob().readall()))
print(df.shape)           # Expected: (80000, 8)
print(df["weather"].value_counts())
print(df["scene"].value_counts())
```

### 5. Validate ADF pipeline runs successfully

```bash
az datafactory pipeline create-run \
  --factory-name trafficprodadf \
  --resource-group rg-traffic-prod \
  --name pipeline_bdd100k_ingestion \
  --parameters '{"storageAccount": "trafficprodlake"}'
```

Monitor in Azure Portal → Data Factory → Monitor → Pipeline Runs.

---

## Pitfalls

| Pitfall | Prevention |
|---------|-----------|
| BDD100K download requires registration | Register at bdd-data.berkeley.edu before starting; download credentials expire |
| Uploading from local machine incurs Azure egress at ~$0.08/GB | Always upload from a VM in the same Azure region as the storage account |
| JSON label files use different schema versions for video vs image labels | The parser targets image labels only; video labels are stored raw for Step 03 |
| `DefaultAzureCredential` fails on VM without managed identity | Assign Storage Blob Data Contributor role to the VM's managed identity |
| ADF pipeline JSON requires linked service names to exist before import | Create linked services in portal before running `az datafactory pipeline create` |
| Parquet `categories` column stores comma-separated strings | Downstream code must split on comma; do not store as a list (Parquet schema compatibility) |

---

## Expected Outcome After Step 02

- `raw-bronze` container holds all BDD100K videos, images, and raw JSON labels — untouched and immutable
- `processed-silver` container holds three Parquet files with 80,000 metadata records covering weather, scene, time-of-day, vehicle counts, and label categories
- Azure Data Factory has a registered, runnable pipeline for the full ingestion flow
- All data access uses `DefaultAzureCredential` — no connection strings in code
- Upload is idempotent — re-running skips already-uploaded files
