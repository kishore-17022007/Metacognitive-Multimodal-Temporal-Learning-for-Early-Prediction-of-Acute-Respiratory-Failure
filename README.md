# Metacognitive Multimodal Temporal Learning for Early Prediction of Acute Respiratory Failure in TBI

## 1. Project Overview

This project presents a multimodal temporal machine learning and MLOps prototype for early prediction of Acute Respiratory Failure (ARF) in patients with Traumatic Brain Injury (TBI).

The research architecture combines a Transformer for learning temporal patterns with a Physiological Graph Attention Network (GAT) for modelling relationships between clinical features. A reliability assessment layer incorporates uncertainty estimation, out-of-distribution (OOD) detection, and prediction confidence.

The project uses synthetic ICU data for model development, evaluation, and cloud deployment demonstrations.

### Machine Learning Workflow

```mermaid
flowchart TD
    A["Synthetic ICU Data"] --> B["Data Preprocessing"]
    B --> C["Temporal Transformer"]
    B --> D["Physiological GAT"]
    C --> E["Gated Fusion and Prediction"]
    D --> E
    E --> F["Reliability Assessment"]
    F --> G["Model Evaluation and Versioning"]
    G --> H["Registered Inference Model"]
```

## 2. AWS Cloud Architecture and Workflow

```mermaid
flowchart TD
    A["Synthetic Data and Model Artifacts"] --> B["Amazon S3"]
    B --> C["AWS Lambda - Inference"]
    D["Client / Dashboard"] --> E["Amazon API Gateway"]
    E --> C
    C --> F["Prediction Response"]
    F --> D
    G["AWS IoT Core"] --> H["IoT Processor Lambda"]
    H --> B
    C --> I["Amazon CloudWatch"]
    H --> I
    I --> J["Amazon SNS Alerts"]
```

This workflow demonstrates cloud storage, serverless inference, API access, IoT data processing, logging, and alert integration. The Transformer-GAT research model is developed separately from the registered CPU-based inference model used by the serving workflow.

## 3. AWS Services and Their Functions

| AWS Service | Purpose in the Project |
|---|---|
| **Amazon S3** | Stores synthetic data, model artifacts, and project files. |
| **AWS Lambda** | Runs serverless inference and IoT processing functions. |
| **Amazon API Gateway** | Provides an API endpoint to access prediction services. |
| **AWS IoT Core** | Supports ingestion of IoT messages and simulated patient data. |
| **Amazon CloudWatch** | Collects execution logs and monitoring information. |
| **Amazon SNS** | Supports notification and alert integration. |
| **Amazon ECR** | Stores container images for cloud deployment workflows. |
| **AWS Amplify** | Supports frontend hosting and dashboard deployment. |
| **AWS IAM** | Manages roles, permissions, and secure service access. |

