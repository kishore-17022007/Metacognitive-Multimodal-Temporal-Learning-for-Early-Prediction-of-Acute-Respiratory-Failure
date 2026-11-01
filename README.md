# Metacognitive Multimodal Temporal Learning for Early Prediction of Acute Respiratory Failure in TBI

A research and MLOps prototype for early prediction of Acute Respiratory Failure (ARF) in Traumatic Brain Injury (TBI) patients using temporal machine learning, physiological feature relationships, and model reliability assessment.

**Dataset:** Synthetic ICU data  
**Purpose:** Academic research and MLOps demonstration only. Not a clinically validated diagnostic tool.

## System Architecture

```mermaid
flowchart TD
    A["Synthetic Patient Data"] --> B["Amazon S3"]
    B --> C["Data Preprocessing & Validation"]

    C --> D["Temporal Transformer"]
    C --> E["Physiological GAT"]

    D --> F["Gated Fusion & Prediction"]
    E --> F

    F --> G["Reliability Assessment"]
    G --> H["Model Artifacts & Registry"]

    H --> I["AWS Lambda"]
    I --> J["Amazon API Gateway"]
    J --> K["Prediction API"]

    K --> L["Streamlit Dashboard"]

    I --> M["Amazon CloudWatch"]
    M --> N["Amazon SNS Alerts"]

    O["AWS IoT Core"] --> I
```

## Technologies Used

- **Machine Learning:** Python, Scikit-learn, PyTorch
- **Backend:** FastAPI, AWS Lambda, API Gateway
- **Cloud Storage:** Amazon S3
- **Monitoring:** Amazon CloudWatch, Amazon SNS
- **IoT Integration:** AWS IoT Core
- **Dashboard:** Streamlit
- **Experiment Tracking:** MLflow

## Key Features

- Temporal patient observation modelling.
- Transformer and physiological GAT-based research architecture.
- Prediction reliability, uncertainty, and OOD assessment.
- Model evaluation, versioning, and experiment tracking.
- AWS-based inference and monitoring demonstration.

