# AI Network Anomaly Detection

An AI-powered network intrusion detection system that classifies network traffic as **Normal** or **Anomalous** using machine learning. The project implements a complete ML pipeline including data preprocessing, feature engineering, model training, model evaluation, explainability with SHAP, and an interactive Streamlit dashboard for real-time predictions.

---

## Features

- 🌐 Detect anomalous network traffic
- 🤖 Machine Learning-based intrusion detection using XGBoost
- 📊 Interactive Streamlit dashboard
- 📈 Model evaluation with Accuracy, Precision, Recall, and F1-score
- 🔍 SHAP-based model explainability
- 📂 Batch prediction using CSV files
- ⚡ Real-time single record prediction
- 🐳 Docker support

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Frontend | Streamlit |
| Deployment | Docker |

---

## Testing

This project has been tested locally on **Windows (Python 3.12)**.

### Verified Components

- ✅ Data preprocessing
- ✅ Feature engineering
- ✅ Feature scaling and encoding
- ✅ Model training
- ✅ XGBoost classifier
- ✅ Model evaluation
- ✅ SHAP explainability
- ✅ Streamlit dashboard
- ✅ Single record prediction
- ✅ Batch prediction using CSV

---

# Project Pipeline

```text
Network Traffic
        │
        ▼
 Data Preprocessing
        │
        ▼
 Feature Engineering
        │
        ▼
 Feature Encoding & Scaling
        │
        ▼
 Train/Test Split
        │
        ▼
 XGBoost Classifier
        │
        ▼
 Model Evaluation
        │
        ▼
 SHAP Explainability
        │
        ▼
 Streamlit Dashboard
        │
        ├──────────────► Single Prediction
        │
        └──────────────► Batch Prediction