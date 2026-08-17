"""
app.py - Telco Customer Churn Classifier
A Streamlit app to explore and compare 5 classification models.
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# ----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Classifier",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .app-header {
        padding: 1.75rem 2rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #1f2937 0%, #111827 100%);
        border: 1px solid #2d3748;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.9rem;
        color: #f9fafb;
    }
    .app-header p {
        margin: 0.4rem 0 0 0;
        color: #9ca3af;
        font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e5e7eb;
        margin: 1.6rem 0 0.6rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #374151;
    }
    div[data-testid="stMetric"] {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 0.9rem 0.6rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af !important;
    }
    div[data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }
    .model-badge {
        display: inline-block;
        background: #1e3a8a;
        color: #bfdbfe;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .footer-note {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2d3748;
        color: #6b7280;
        font-size: 0.82rem;
        text-align: center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}
SCALED_MODELS = {"Logistic Regression", "kNN"}

MODEL_BLURBS = {
    "Logistic Regression": "A linear baseline — fast, interpretable, and strong when churn drivers behave roughly linearly.",
    "Decision Tree": "Splits customers by simple if/else rules on features like contract type and tenure.",
    "kNN": "Classifies a customer based on the churn outcome of its nearest neighbors in feature space.",
    "Naive Bayes": "Assumes features are independent — fast, but usually the highest-recall / lowest-precision model here.",
    "Random Forest (Ensemble)": "An ensemble of decision trees — typically the best balance of precision and recall.",
}


@st.cache_resource
def load_model(name):
    with open(MODEL_FILES[name], "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_columns():
    with open("model/feature_columns.json") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📡 Telco Customer Churn — Classification Explorer</h1>
        <p>Upload test data, choose a model, and compare performance across 6 evaluation metrics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_cols = load_feature_columns()
scaler = load_scaler()

# ----------------------------------------------------------------------------
# SIDEBAR — controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    use_default = st.checkbox("Use bundled sample test_data.csv", value=True)
    uploaded_file = None
    if not use_default:
        uploaded_file = st.file_uploader(
            "Upload test CSV (must include 'Target' column)", type=["csv"]
        )

    st.markdown("---")
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    st.caption(MODEL_BLURBS[model_name])

    st.markdown("---")
    st.markdown(
        "**About this dataset**\n\n"
        "7,043 telecom customers, 30 engineered features "
        "(demographics, account info, subscribed services). "
        "Target: whether the customer churned."
    )

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
if use_default:
    data = pd.read_csv("test_data.csv")
elif uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("👈 Upload a CSV in the sidebar, or check 'Use bundled sample test_data.csv'.")
    st.stop()

missing_cols = [c for c in feature_cols if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required feature columns: {missing_cols}")
    st.stop()
if "Target" not in data.columns:
    st.error("Uploaded file must include a 'Target' column (0 = No Churn, 1 = Churn).")
    st.stop()

X = data[feature_cols]
y_true = data["Target"]

# ----------------------------------------------------------------------------
# DATASET PREVIEW
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">📁 Dataset Preview</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{data.shape[0]:,}")
c2.metric("Features", f"{len(feature_cols)}")
c3.metric("Churn Rate", f"{y_true.mean():.1%}")
with st.expander("View raw data sample"):
    st.dataframe(data.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------------------------------
model = load_model(model_name)
X_eval = scaler.transform(X) if model_name in SCALED_MODELS else X
y_pred = model.predict(X_eval)
y_proba = model.predict_proba(X_eval)[:, 1]

# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
st.markdown(f'<span class="model-badge">Selected model: {model_name}</span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Evaluation Metrics</div>', unsafe_allow_html=True)

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_proba),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC": matthews_corrcoef(y_true, y_pred),
}
cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, f"{value:.3f}")

# ----------------------------------------------------------------------------
# CONFUSION MATRIX + CLASSIFICATION REPORT
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">🔍 Confusion Matrix & Classification Report</div>', unsafe_allow_html=True)
left, right = st.columns([1, 1.3])

with left:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        cbar=False, ax=ax, annot_kws={"size": 13},
    )
    ax.set_xlabel("Predicted", color="#374151")
    ax.set_ylabel("Actual", color="#374151")
    ax.tick_params(colors="#374151")
    st.pyplot(fig, use_container_width=True)

with right:
    report = classification_report(
        y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True
    )
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ----------------------------------------------------------------------------
# MODEL COMPARISON
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">🏆 Compare All Models</div>', unsafe_allow_html=True)
show_comparison = st.checkbox("Show comparison across all 5 models", value=False)
if show_comparison:
    summary = pd.read_csv("model/metrics_summary.csv")
    st.dataframe(
        summary.style.highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#1e3a8a",
        ),
        use_container_width=True,
    )
    st.bar_chart(summary.set_index("Model")[["Accuracy", "AUC", "F1", "MCC"]])

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
        Built for the BITS Pilani M.Tech (AIML/DSE) Machine Learning Assignment 2 ·
        6 classification models, evaluated on Accuracy, AUC, Precision, Recall, F1, and MCC.
    </div>
    """,
    unsafe_allow_html=True,
)
