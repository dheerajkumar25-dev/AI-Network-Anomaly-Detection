# app.py
# Streamlit web app for the AI Network Anomaly Detection project.
#
# Run from the project root:
#   streamlit run app/app.py

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st
from predict import predict_single, predict_batch, explain_single, SHAP_AVAILABLE

st.set_page_config(page_title="Network Anomaly Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Network Anomaly Detection")
st.write("Detects suspicious network connections (DoS floods, port scans, etc.) using a trained ML model.")

# ---------- Dataset dashboard cards ----------
summary_path = "reports/dataset_summary.json"
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{summary['total_records']:,}")
    c2.metric("Normal Traffic", f"{summary['normal_count']:,}")
    c3.metric("Anomalies", f"{summary['anomaly_count']:,}")
    c4.metric("Attack %", f"{summary['attack_percentage']}%")
    st.divider()

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction (CSV)"])

# ---------- TAB 1: single record ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        duration = st.number_input("Duration (seconds)", min_value=0.0, value=1.0)
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
        service = st.selectbox("Service", ["http", "ftp", "smtp", "other"])
        flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO"])
        src_bytes = st.number_input("Source Bytes", min_value=0.0, value=300.0)

    with col2:
        dst_bytes = st.number_input("Destination Bytes", min_value=0.0, value=500.0)
        count = st.number_input("Connection Count", min_value=0.0, value=5.0)
        srv_count = st.number_input("Service Count", min_value=0.0, value=5.0)
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.9)

    record = {
        "duration": duration, "protocol_type": protocol_type, "service": service,
        "flag": flag, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "count": count, "srv_count": srv_count, "serror_rate": serror_rate,
        "same_srv_rate": same_srv_rate,
    }

    if st.button("Analyze Traffic", type="primary"):
        try:
            result = predict_single(record)

            # Colored result card
            if result["prediction"] == "normal":
                st.success(f"✅ NORMAL traffic — confidence {result['confidence']*100:.2f}%")
            else:
                st.error(f"🚨 ANOMALY detected — confidence {result['confidence']*100:.2f}%")

            left, right = st.columns(2)

            with left:
                st.subheader("Prediction probability")
                proba_df = pd.DataFrame({
                    "class": ["normal", "anomaly"],
                    "probability": [result["prob_normal"], result["prob_anomaly"]],
                }).set_index("class")
                st.bar_chart(proba_df)

            with right:
                st.subheader("Why this prediction?")
                if not SHAP_AVAILABLE:
                    st.caption("Using feature-importance fallback (install `shap` for per-prediction SHAP values).")
                explanation = explain_single(record)
                for feature, value in explanation:
                    direction = "→ pushes toward ANOMALY" if value > 0 else "→ pushes toward NORMAL"
                    st.write(f"**{feature}**: {value:+.3f}  {direction}")

            st.subheader("Input summary")
            st.table(pd.DataFrame([record]).T.rename(columns={0: "value"}))

        except FileNotFoundError:
            st.warning("Model not found. Run 'python src/preprocess.py' then 'python src/train.py' first.")

# ---------- TAB 2: batch CSV ----------
with tab2:
    st.write(
        "Upload a CSV with columns: `duration, protocol_type, service, flag, "
        "src_bytes, dst_bytes, count, srv_count, serror_rate, same_srv_rate`"
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            result_df = predict_batch(input_df)

            st.write(f"Predicted {len(result_df)} rows.")
            n_anomaly = (result_df["prediction"] == "anomaly").sum()
            st.write(f"Flagged **{n_anomaly}** as anomaly out of {len(result_df)}.")

            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not process this file: {e}")

st.divider()
st.caption(
    "Tip for the single-prediction tab: try duration=0.01, protocol=icmp, service=other, "
    "flag=S0, count=350, serror_rate=0.9 to see an anomaly get flagged."
)
