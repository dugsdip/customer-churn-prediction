"""
Streamlit app: enter customer details -> get churn probability.

Run:
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_DIR / "churn_model.joblib")
    return bundle["pipeline"], bundle["threshold"]


pipeline, threshold = load_model()

st.title("📉 Customer Churn Predictor")
st.caption(
    "Enter a telecom customer's details to estimate the probability that "
    "they will leave. Model: tuned XGBoost trained on the Telco Customer Churn dataset."
)

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
    with c2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0, step=0.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with c3:
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        phone = st.selectbox("Phone service", ["Yes", "No"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

    # Add-on services only make sense when the base service exists
    multi_lines = "No phone service"
    if phone == "Yes":
        multi_lines = st.selectbox("Multiple lines", ["No", "Yes"])

    addons = {}
    addon_names = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
    if internet == "No":
        for a in addon_names:
            addons[a] = "No internet service"
    else:
        st.markdown("**Internet add-ons**")
        cols = st.columns(3)
        for i, a in enumerate(addon_names):
            label = {"OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
                     "DeviceProtection": "Device protection", "TechSupport": "Tech support",
                     "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies"}[a]
            addons[a] = cols[i % 3].selectbox(label, ["No", "Yes"], key=a)

row = {
    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multi_lines, "InternetService": internet,
    **addons,
    "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": round(tenure * monthly, 2),
}
X_new = pd.DataFrame([row])

with right:
    st.subheader("Prediction")
    prob = float(pipeline.predict_proba(X_new)[0, 1])
    will_churn = prob >= threshold

    st.metric("Churn probability", f"{prob:.1%}")
    st.progress(min(prob, 1.0))
    if will_churn:
        st.error("⚠️ High risk — this customer is likely to churn.")
        st.write("Consider a retention offer, a contract upgrade discount, or a proactive call.")
    elif prob >= threshold * 0.6:
        st.warning("Medium risk — worth keeping an eye on.")
    else:
        st.success("✅ Low risk — this customer is likely to stay.")
    st.caption(f"Decision threshold used by the model: {threshold:.2f}")

st.divider()
with st.expander("About the model"):
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())["test_metrics"]
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        b.metric("Recall", f"{m['recall']:.3f}")
        c.metric("Precision", f"{m['precision']:.3f}")
        d.metric("F1", f"{m['f1']:.3f}")
    res = MODEL_DIR / "results.csv"
    if res.exists():
        st.dataframe(pd.read_csv(res), hide_index=True)
    p1, p2 = MODEL_DIR / "evaluation.png", MODEL_DIR / "feature_importance.png"
    if p1.exists() and p2.exists():
        i1, i2 = st.columns(2)
        i1.image(str(p1))
        i2.image(str(p2))
