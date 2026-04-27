# Step 01 — Azure Infrastructure Bootstrap

## Objective

Provision every Azure resource the project depends on before any code or data pipeline is written. This creates the foundation: networking, storage, identity, and compute namespaces that all later steps reference. Getting infrastructure right here prevents costly rework in every subsequent sprint.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Azure Subscription                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               Resource Group: rg-traffic-prod               │   │
│  │                                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│  │  │ Azure Blob   │  │  Azure Event │  │   Azure ML       │  │   │
│  │  │ Storage      │  │  Hubs NS     │  │   Workspace      │  │   │
│  │  │ (Data Lake)  │  │              │  │                  │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │   │
│  │                                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│  │  │  Azure       │  │  Azure       │  │  Azure           │  │   │
│  │  │  Databricks  │  │  Data        │  │  Kubernetes      │  │   │
│  │  │  Workspace   │  │  Factory     │  │  Service (AKS)   │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │   │
│  │                                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐                        │   │
│  │  │  Azure       │  │  Azure       │                        │   │
│  │  │  Container   │  │  Key Vault   │                        │   │
│  │  │  Registry    │  │              │                        │   │
│  │  └──────────────┘  └──────────────┘                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Azure CLI | >= 2.55 | All resource provisioning |
| kubectl | >= 1.28 | AKS interaction |
| Helm | >= 3.12 | AKS chart deployments |
| Python | >= 3.11 | SDK scripts |
| Git | any | Version control |

Install Azure CLI:
```bash
brew install azure-cli          # macOS
# or
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash   # Ubuntu
```

---

## Setup Instructions

### 1. Login and Set Subscription

```bash
az login
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
az account show   # verify
```

### 2. Define Environment Variables (save to infra/.env)

```bash
export LOCATION="eastus2"
export RG="rg-traffic-prod"
export PREFIX="trafficprod"
export STORAGE_ACCOUNT="${PREFIX}dlake"
export EVENTHUB_NS="${PREFIX}ehns"
export AML_WORKSPACE="${PREFIX}aml"
export ADB_WORKSPACE="${PREFIX}adb"
export ADF_NAME="${PREFIX}adf"
export AKS_NAME="${PREFIX}aks"
export ACR_NAME="${PREFIX}acr"
export KV_NAME="${PREFIX}kv"
export AKS_NODE_COUNT=3
export AKS_VM_SIZE="Standard_DS3_v2"
```

### 3. Create Resource Group

```bash
az group create \
  --name $RG \
  --location $LOCATION \
  --tags project=SmartTrafficMonitoring env=prod
```

### 4. Create Azure Data Lake Storage Gen2

```bash
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RG \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2 \
  --hierarchical-namespace true \
  --access-tier Hot

# Create containers (zones)
for CONTAINER in raw-bronze processed-silver gold-serving models checkpoints; do
  az storage container create \
    --account-name $STORAGE_ACCOUNT \
    --name $CONTAINER \
    --auth-mode login
done
```

**Data Lake Zones:**
| Container | Purpose |
|-----------|---------|
| `raw-bronze` | Raw BDD100K video, images, JSON metadata — never modified |
| `processed-silver` | Extracted frames, normalized metadata, cleaned data |
| `gold-serving` | Inference-ready tensors, aggregated analytics |
| `models` | Exported model artifacts (ONNX, PT, TensorRT) |
| `checkpoints` | Training checkpoints per experiment |

### 5. Create Azure Event Hubs Namespace + Hub

```bash
az eventhubs namespace create \
  --name $EVENTHUB_NS \
  --resource-group $RG \
  --location $LOCATION \
  --sku Standard \
  --capacity 4

az eventhubs eventhub create \
  --name traffic-stream \
  --namespace-name $EVENTHUB_NS \
  --resource-group $RG \
  --partition-count 16 \
  --message-retention 1
```

### 6. Create Azure Machine Learning Workspace

```bash
az ml workspace create \
  --name $AML_WORKSPACE \
  --resource-group $RG \
  --location $LOCATION \
  --storage-account $STORAGE_ACCOUNT \
  --sku Basic
```

### 7. Create Azure Databricks Workspace

```bash
az databricks workspace create \
  --name $ADB_WORKSPACE \
  --resource-group $RG \
  --location $LOCATION \
  --sku standard
```

### 8. Create Azure Data Factory

```bash
az datafactory factory create \
  --factory-name $ADF_NAME \
  --resource-group $RG \
  --location $LOCATION
```

### 9. Create Azure Container Registry

```bash
az acr create \
  --name $ACR_NAME \
  --resource-group $RG \
  --location $LOCATION \
  --sku Premium \
  --admin-enabled false
```

### 10. Create Azure Kubernetes Service

```bash
az aks create \
  --name $AKS_NAME \
  --resource-group $RG \
  --location $LOCATION \
  --node-count $AKS_NODE_COUNT \
  --node-vm-size $AKS_VM_SIZE \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --attach-acr $ACR_NAME \
  --network-plugin azure \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10

# Add GPU node pool for inference
az aks nodepool add \
  --cluster-name $AKS_NAME \
  --resource-group $RG \
  --name gpupool \
  --node-count 1 \
  --node-vm-size Standard_NC6s_v3 \
  --node-taints sku=gpu:NoSchedule \
  --labels accelerator=nvidia

# Fetch credentials
az aks get-credentials \
  --name $AKS_NAME \
  --resource-group $RG
```

### 11. Create Azure Key Vault

```bash
az keyvault create \
  --name $KV_NAME \
  --resource-group $RG \
  --location $LOCATION \
  --enable-rbac-authorization true

# Store storage connection string as secret
CONN_STR=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT --resource-group $RG --query connectionString -o tsv)

az keyvault secret set \
  --vault-name $KV_NAME \
  --name storage-connection-string \
  --value "$CONN_STR"
```

### 12. Create Service Principal for MLOps CI/CD

```bash
SP_JSON=$(az ad sp create-for-rbac \
  --name "sp-traffic-mlops" \
  --role Contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/$RG \
  --sdk-auth)

echo "$SP_JSON"
# Save this output — it is used in Azure DevOps pipeline secrets
```

---

## Project Folder Structure (Final Target)

```
Traffic Monitoring/
├── infra/
│   ├── .env                          # Environment variables (gitignored)
│   ├── provision.sh                  # Master provisioning script
│   └── teardown.sh                   # Cleanup script
├── steps/
│   ├── step-01-azure-infrastructure-bootstrap/
│   │   └── README.md
│   ├── step-02-data-ingestion-bdd100k/
│   ├── step-03-databricks-frame-processing/
│   ├── step-04-yolov8-training-azureml/
│   ├── step-05-deepsort-tracking/
│   ├── step-06-convlstm-accident-detection/
│   ├── step-07-mlops-model-registry/
│   ├── step-08-aks-realtime-inference/
│   ├── step-09-batch-pipeline/
│   ├── step-10-powerbi-dashboard/
│   ├── step-11-cicd-azure-devops/
│   └── step-12-monitoring-logging/
├── src/
│   ├── ingestion/                    # BDD100K download + upload scripts
│   ├── processing/                   # Databricks notebooks
│   ├── training/
│   │   ├── yolov8/
│   │   ├── deepsort/
│   │   └── convlstm/
│   ├── inference/
│   │   ├── realtime/                 # Event Hubs consumer + AKS serving
│   │   └── batch/                    # ADF + Databricks batch
│   ├── api/                          # FastAPI inference endpoints
│   └── dashboard/                    # Power BI dataset connectors
├── deploy/
│   ├── aks/                          # Kubernetes manifests
│   └── aml/                          # Azure ML pipeline YAML
├── .github/
│   └── workflows/                    # CI/CD pipelines
├── .gitignore
└── README.md
```

---

## infra/provision.sh

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/.env"

echo "[1/11] Creating Resource Group..."
az group create --name $RG --location $LOCATION --tags project=SmartTrafficMonitoring env=prod

echo "[2/11] Creating Storage Account (ADLS Gen2)..."
az storage account create --name $STORAGE_ACCOUNT --resource-group $RG \
  --location $LOCATION --sku Standard_LRS --kind StorageV2 \
  --hierarchical-namespace true --access-tier Hot

for CONTAINER in raw-bronze processed-silver gold-serving models checkpoints; do
  az storage container create --account-name $STORAGE_ACCOUNT --name $CONTAINER --auth-mode login
done

echo "[3/11] Creating Event Hubs..."
az eventhubs namespace create --name $EVENTHUB_NS --resource-group $RG \
  --location $LOCATION --sku Standard --capacity 4
az eventhubs eventhub create --name traffic-stream \
  --namespace-name $EVENTHUB_NS --resource-group $RG \
  --partition-count 16 --message-retention 1

echo "[4/11] Creating Azure ML Workspace..."
az ml workspace create --name $AML_WORKSPACE --resource-group $RG \
  --location $LOCATION --storage-account $STORAGE_ACCOUNT

echo "[5/11] Creating Databricks Workspace..."
az databricks workspace create --name $ADB_WORKSPACE --resource-group $RG \
  --location $LOCATION --sku standard

echo "[6/11] Creating Data Factory..."
az datafactory factory create --factory-name $ADF_NAME --resource-group $RG --location $LOCATION

echo "[7/11] Creating Container Registry..."
az acr create --name $ACR_NAME --resource-group $RG \
  --location $LOCATION --sku Premium --admin-enabled false

echo "[8/11] Creating AKS Cluster..."
az aks create --name $AKS_NAME --resource-group $RG \
  --location $LOCATION --node-count $AKS_NODE_COUNT \
  --node-vm-size $AKS_VM_SIZE --enable-addons monitoring \
  --generate-ssh-keys --attach-acr $ACR_NAME \
  --network-plugin azure --enable-cluster-autoscaler \
  --min-count 2 --max-count 10

az aks nodepool add --cluster-name $AKS_NAME --resource-group $RG \
  --name gpupool --node-count 1 --node-vm-size Standard_NC6s_v3 \
  --node-taints sku=gpu:NoSchedule --labels accelerator=nvidia

az aks get-credentials --name $AKS_NAME --resource-group $RG

echo "[9/11] Creating Key Vault..."
az keyvault create --name $KV_NAME --resource-group $RG \
  --location $LOCATION --enable-rbac-authorization true

CONN_STR=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT --resource-group $RG --query connectionString -o tsv)
az keyvault secret set --vault-name $KV_NAME \
  --name storage-connection-string --value "$CONN_STR"

echo "[10/11] Creating Service Principal..."
az ad sp create-for-rbac --name "sp-traffic-mlops" --role Contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RG \
  --sdk-auth

echo "[11/11] Done. All resources provisioned in $RG."
```

---

## Validation Procedure

Run these commands to confirm every resource exists and is healthy:

```bash
# 1. List all resources in the group
az resource list --resource-group $RG --output table

# 2. Verify storage containers
az storage container list --account-name $STORAGE_ACCOUNT --auth-mode login --output table

# 3. Verify Event Hub
az eventhubs eventhub show \
  --name traffic-stream \
  --namespace-name $EVENTHUB_NS \
  --resource-group $RG --query "{name:name,partitions:partitionCount,status:status}"

# 4. Verify AML workspace
az ml workspace show --name $AML_WORKSPACE --resource-group $RG --query "{name:name,location:location}"

# 5. Verify AKS nodes
kubectl get nodes -o wide

# 6. Verify ACR
az acr show --name $ACR_NAME --resource-group $RG --query loginServer

# 7. Verify Key Vault secret
az keyvault secret show --vault-name $KV_NAME --name storage-connection-string --query "value" -o tsv | head -c 50
```

**Expected:** All resources show `Succeeded` provisioning state. AKS has 3 standard nodes + 1 GPU node. Storage has 5 containers.

---

## Pitfalls

| Pitfall | Prevention |
|---------|-----------|
| Storage account name must be globally unique (3-24 lowercase alphanumeric) | Use `${PREFIX}dlake` with a unique prefix |
| AKS GPU node pool requires NC-series quota approval | Submit quota request to Azure Support before this step |
| Key Vault soft-delete cannot be immediately re-used if name was used before | Use a unique KV name per environment |
| `az ml workspace create` fails if storage account is in different region | Always use the same `$LOCATION` for all resources |
| Service Principal secret expires in 1 year by default | Set `--years 2` or rotate via DevOps pipeline |
| ACR Premium tier required for geo-replication and content trust | Do not downgrade to Basic |

---

## Expected Outcome After Step 01

- All 8 Azure services are provisioned and healthy in `rg-traffic-prod`
- Data Lake Gen2 has 5 containers with correct zone structure
- AKS cluster has auto-scaling enabled with a dedicated GPU node pool
- All secrets are stored in Key Vault — no credentials in code
- Service Principal is created for CI/CD pipelines
- `infra/provision.sh` is reproducible and idempotent
