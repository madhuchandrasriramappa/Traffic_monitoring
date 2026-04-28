# Smart Traffic Monitoring System (Azure)

Production-grade computer vision system for real-time and batch traffic analysis on Microsoft Azure.

## Capabilities
- Vehicle detection (YOLOv8, fine-tuned on BDD100K)
- Multi-object tracking (DeepSORT)
- Road accident detection (ConvLSTM spatio-temporal anomaly detection)
- Real-time streaming inference via Azure Event Hubs + AKS
- Batch processing via ADF + Databricks
- Live Power BI analytics dashboard

## Architecture

```
BDD100K Dataset
      │
      ▼
Azure Blob Storage (ADLS Gen2)          ← Data Lake: Bronze / Silver / Gold
      │
      ▼
Azure Data Factory                      ← Pipeline Orchestration
      │
      ├──► Azure Databricks             ← Distributed Frame Processing
      │         │
      │         ▼
      │    Processed Frames (Silver)
      │
      ▼
Azure Machine Learning
      ├── YOLOv8 Fine-tuning
      ├── ConvLSTM Training
      ├── MLflow Experiment Tracking
      └── Model Registry
            │
            ▼
Azure Kubernetes Service (AKS)         ← Real-Time Inference
      │
      ├──► Azure Event Hubs            ← Stream Ingestion
      └──► Batch (ADF trigger)
            │
            ▼
      Power BI Dashboard
```

## Sprint Steps

| Step | Name | Status |
|------|------|--------|
| 01 | Azure Infrastructure Bootstrap | ✅ |
| 02 | Data Ingestion (COCO128) | ✅ |
| 03 | Databricks Frame Processing | ✅ |
| 04 | YOLOv8 Training on Azure ML | ✅ |
| 05 | DeepSORT Object Tracking | ✅ |
| 06 | ConvLSTM Accident Detection | ✅ |
| 07 | MLOps Model Registry | 🔜 |
| 08 | AKS Real-Time Inference | 🔜 |
| 09 | Batch Processing Pipeline | 🔜 |
| 10 | Power BI Dashboard | 🔜 |
| 11 | CI/CD with Azure DevOps | 🔜 |
| 12 | Monitoring & Logging | 🔜 |

## Quickstart

```bash
cp infra/.env.example infra/.env
# Edit infra/.env with your subscription details
chmod +x infra/provision.sh
./infra/provision.sh
```
