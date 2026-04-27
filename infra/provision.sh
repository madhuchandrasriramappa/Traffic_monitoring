#!/usr/bin/env bash
# Master provisioning script — Smart Traffic Monitoring System
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
