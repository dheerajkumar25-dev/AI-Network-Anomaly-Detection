# AI Network Anomaly Detection

A machine learning project that classifies network connections as **normal** or **anomaly** (e.g. DoS flood, port scan) using traffic statistics like connection count, byte counts, and error rates. Includes a full pipeline (preprocessing → training → evaluation), an **XGBoost** classifier with **SHAP explainability**, and a Streamlit app with a metrics dashboard, single-record prediction, and CSV batch prediction.

## About the dataset (please read this before your interview)

This project is built around the column structure of **NSL-KDD**, a well-known public intrusion-detection dataset. However:

- **By default, the code trains on a synthetic dataset** that I generate locally, using the same 10 features and the same normal/anomaly structure as NSL-KDD. I could not auto-download the real file into this environment (GitHub blocks scripted downloads of it, Kaggle needs a login).
- **You should download the real dataset yourself** before using this for a placement/resume claim. It's free, no special access needed:
  - Kaggle: search "NSL-KDD" (e.g. `kaggle.com/datasets/hassan06/nslkdd`)
  - Or the original source: `unb.ca/cic/datasets/nsl.html`
- Save the file as `data/raw/KDDTrain+.txt`. The very next time you run `python src/preprocess.py`, it automatically detects the real file and uses it instead of synthetic data — no code changes needed. It prints clearly which one it used, so you always know.

**Be honest in interviews**: if you haven't swapped in the real file, say you built the pipeline against NSL-KDD's schema and trained on a synthetic stand-in dataset — that's still a legitimate, explainable project. Don't claim "trained on NSL-KDD" unless you've actually run it on the real file.

## Project Structure

```
AI-Network-Anomaly-Detection/
├── data/
│   ├── raw/                 # place KDDTrain+.txt here for real data (see above)
│   └── processed/           # cleaned + encoded + scaled data (auto-generated)
├── notebooks/
│   └── EDA.ipynb            # data exploration
├── src/
│   ├── utils.py             # shared feature-column list
│   ├── preprocess.py        # load/clean/encode/scale data
│   ├── train.py              # train XGBoost (or RF fallback), save model + metrics
│   └── predict.py            # load model, predict single/batch, explain predictions
├── models/
│   ├── model.pkl               # latest trained model (auto-generated)
│   ├── encoders.pkl            # (auto-generated)
│   ├── scaler.pkl               # (auto-generated)
│   ├── version_history.json     # accuracy/precision/recall per training run
│   └── versions/                 # timestamped model copies
├── app/
│   └── app.py                 # Streamlit app (dashboard, single + batch prediction)
├── reports/
│   ├── images/                 # confusion matrix, feature importance
│   ├── metrics.txt             # accuracy / precision / recall / F1
│   └── dataset_summary.json    # record counts for the dashboard cards
├── requirements.txt
├── Dockerfile
└── README.md
```

## How it works

1. **`src/preprocess.py`** — loads the data (real NSL-KDD if present, otherwise synthetic), cleans missing values, encodes text columns (`protocol_type`, `service`, `flag`) into numbers, scales numeric columns, and saves dataset stats for the dashboard.
2. **`src/train.py`** — splits data 80/20, trains an **XGBoost** classifier (falls back to Random Forest automatically if `xgboost` isn't installed — the pipeline never breaks), evaluates it, and saves the model, a metrics report, two plots, and a versioned copy in `models/versions/`.
3. **`src/predict.py`** — loads the saved model and exposes:
   - `predict_single(record)` — classify one connection
   - `predict_batch(df)` — classify a whole CSV at once
   - `explain_single(record)` — top features driving that prediction, using **SHAP** if installed (falls back to feature importances otherwise)
4. **`app/app.py`** — a Streamlit app with:
   - dashboard cards (total records, normal traffic, anomalies, attack %)
   - a single-prediction tab with a colored result card, probability chart, and SHAP-based explanation
   - a batch-prediction tab: upload a CSV, get predictions for every row, download the results

### Model versioning

Every time you run `train.py`, it saves a timestamped copy to `models/versions/` and appends a record (model type, accuracy, precision, recall, F1) to `models/version_history.json`. `models/model.pkl` always points to the latest one — that's what the app loads.


## Results (on the synthetic dataset)

I couldn't install `xgboost` while preparing this project (no internet in the build sandbox), so the numbers shipped here were generated using the **Random Forest fallback** — you'll see this noted in `reports/metrics.txt`. Once you `pip install -r requirements.txt` on your own machine, `train.py` will automatically use **XGBoost** instead. Re-run it and update this table with your real numbers.

| Metric | Score (Random Forest, this build) |
|---|---|
| Accuracy | 98.0% |
| Precision | 98.5% |
| Recall | 92.0% |
| F1-score | 95.1% |

Full report: [`reports/metrics.txt`](reports/metrics.txt).

![Confusion Matrix](reports/images/confusion_matrix.png)
![Feature Importance](reports/images/feature_importance.png)

`serror_rate`, `flag`, and `count` are consistently the strongest predictors — this matches how real DoS/probe attacks look (many failed or rapid-fire connections).

## Getting Started

```bash
git clone https://github.com/<your-username>/AI-Network-Anomaly-Detection.git
cd AI-Network-Anomaly-Detection
pip install -r requirements.txt

# run every script from the project root, not from inside src/
python src/preprocess.py
python src/train.py
python src/predict.py       # quick terminal demo

streamlit run app/app.py    # opens the web demo in your browser
```

### Run with Docker

```bash
docker build -t network-anomaly-detector .
docker run -p 8501:8501 network-anomaly-detector
```

## Possible next steps

- Train on the real NSL-KDD dataset (see above) or CICIDS2017 for stronger, resume-ready numbers.
- Try an unsupervised model (Isolation Forest) to catch attack types the model has never seen labeled examples of.
- Add a `/predict` JSON API endpoint for integration with other tools.
- Add unit tests for `preprocess.py` and `predict.py`.
- Add authentication if you deploy the batch-upload feature publicly (right now anyone with the link can upload data).
