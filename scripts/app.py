import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, KNeighborsClassifier

# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("Anomaly Detection Dashboard")

# =====================================================
# Upload
# =====================================================
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# Label column
# =====================================================
st.sidebar.header("2. Label Column")
label_col = st.sidebar.selectbox("Select label column", df.columns)

# =====================================================
# Basic preprocessing ONLY
# =====================================================
st.sidebar.header("3. Basic Preparation")

apply_preprocess = st.sidebar.button("Prepare Data")

def preprocess(df):
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    if not set(y.unique()).issubset({0, 1}):
        st.error("Label column must be binary (0 = normal, 1 = anomaly)")
        st.stop()

    return X, y

if apply_preprocess:
    X, y = preprocess(df)
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.ready = True

if "ready" not in st.session_state:
    st.warning("Click 'Prepare Data' to continue.")
    st.stop()

X = st.session_state.X
y = st.session_state.y

st.success("Data prepared")

# =====================================================
# Train/Test Split
# =====================================================
st.sidebar.header("4. Train/Test Split")

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=random_state
)

# =====================================================
# Missing Values (AFTER SPLIT)
# =====================================================
st.sidebar.header("5. Missing Values")

missing_strategy = st.sidebar.selectbox(
    "Handling method",
    ["None", "Drop rows", "Fill mean", "Fill median"]
)

def apply_missing(X_tr, y_tr, X_te, y_te):
    if missing_strategy == "Drop rows":
        train_mask = X_tr.notna().all(axis=1)
        test_mask = X_te.notna().all(axis=1)

        return (
            X_tr[train_mask], y_tr[train_mask],
            X_te[test_mask], y_te[test_mask]
        )

    elif missing_strategy == "Fill mean":
        m = X_tr.mean()
        return X_tr.fillna(m), y_tr, X_te.fillna(m), y_te

    elif missing_strategy == "Fill median":
        m = X_tr.median()
        return X_tr.fillna(m), y_tr, X_te.fillna(m), y_te

    return X_tr, y_tr, X_te, y_te

X_train, y_train, X_test, y_test = apply_missing(X_train, y_train, X_test, y_test)

# =====================================================
# Scaling
# =====================================================
st.sidebar.header("6. Scaling")

scaling_method = st.sidebar.selectbox(
    "Select scaling method",
    ["None", "Standard", "Robust", "MinMax"]
)

def apply_scaling(X_tr, X_te):
    if scaling_method == "Standard":
        sc = StandardScaler()
    elif scaling_method == "Robust":
        sc = RobustScaler()
    elif scaling_method == "MinMax":
        sc = MinMaxScaler()
    else:
        return X_tr, X_te

    return sc.fit_transform(X_tr), sc.transform(X_te)

# =====================================================
# Models
# =====================================================
st.sidebar.header("7. Models")

models = st.sidebar.multiselect(
    "Select models",
    ["Isolation Forest", "LOF", "k-NN", "Random Forest"],
    default=["Isolation Forest"]
)

train_on_normal_only = st.sidebar.checkbox(
    "Train anomaly models on NORMAL class only", True
)

if scaling_method == "None" and "k-NN" in models:
    st.warning("⚠️ k-NN performs better with scaling.")

# =====================================================
# Hyperparameters
# =====================================================
st.sidebar.header("8. Hyperparameters")
params = {}

if "Isolation Forest" in models:
    params["Isolation Forest"] = {
        "n_estimators": st.sidebar.slider("IF n_estimators", 50, 300, 100),
        "contamination": st.sidebar.slider("IF contamination", 0.01, 0.2, 0.05)
    }

if "LOF" in models:
    params["LOF"] = {
        "n_neighbors": st.sidebar.slider("LOF neighbors", 5, 50, 20)
    }

if "k-NN" in models:
    params["k-NN"] = {
        "n_neighbors": st.sidebar.slider("k-NN neighbors", 3, 50, 10),
        "mode": st.sidebar.selectbox("k-NN mode", ["distance", "classifier"])
    }

if "Random Forest" in models:
    params["Random Forest"] = {
        "n_estimators": st.sidebar.slider("RF n_estimators", 50, 300, 100),
        "max_depth": st.sidebar.slider("RF max_depth", 3, 20, 10)
    }

# =====================================================
# Cross Validation
# =====================================================
st.sidebar.header("9. Validation")

use_cv = st.sidebar.checkbox("Use Cross Validation", False)
cv_folds = st.sidebar.slider("CV folds", 3, 10, 5)

run = st.sidebar.button("Run Experiments")

# =====================================================
# Evaluation
# =====================================================
def evaluate(name, p):
    start = time.time()

    def train_and_score(X_tr, y_tr, X_te, y_te):
        if train_on_normal_only and name in ["Isolation Forest", "LOF"]:
            X_fit = X_tr[y_tr == 0]
        else:
            X_fit = X_tr

        if name == "Isolation Forest":
            model = IsolationForest(**p, random_state=42)
            model.fit(X_fit)
            scores = -model.decision_function(X_te)

        elif name == "LOF":
            model = LocalOutlierFactor(n_neighbors=p["n_neighbors"], novelty=True)
            model.fit(X_fit)
            scores = -model.decision_function(X_te)

        elif name == "k-NN":
            if p["mode"] == "distance":
                model = NearestNeighbors(n_neighbors=p["n_neighbors"])
                model.fit(X_fit)
                d, _ = model.kneighbors(X_te)
                scores = d.mean(axis=1)
            else:
                model = KNeighborsClassifier(n_neighbors=p["n_neighbors"])
                model.fit(X_tr, y_tr)
                scores = model.predict_proba(X_te)[:, 1]

        elif name == "Random Forest":
            model = RandomForestClassifier(**p, random_state=42)
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_te)[:, 1]

        return roc_auc_score(y_te, scores), scores

    # ===== CV =====
    if use_cv:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        aucs = []

        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            X_tr, y_tr, X_te, y_te = apply_missing(X_tr, y_tr, X_te, y_te)
            X_tr, X_te = apply_scaling(X_tr, X_te)

            auc, _ = train_and_score(X_tr, y_tr, X_te, y_te)
            aucs.append(auc)

        return np.mean(aucs), time.time() - start, None, None

    # ===== Normal =====
    else:
        X_tr, X_te = apply_scaling(X_train.copy(), X_test.copy())
        auc, scores = train_and_score(X_tr, y_train, X_te, y_test)
        fpr, tpr, _ = roc_curve(y_test, scores)

        return auc, time.time() - start, fpr, tpr

# =====================================================
# Run
# =====================================================
if run:
    st.subheader("Results")

    rows = []
    fig, ax = plt.subplots()

    for m in models:
        roc_auc, t, fpr, tpr = evaluate(m, params[m])

        rows.append({
            "Model": m,
            "ROC AUC": round(roc_auc, 4),
            "Time (s)": round(t, 3)
        })

        if not use_cv:
            ax.plot(fpr, tpr, label=m)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if not use_cv:
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ROC curve disabled when using Cross Validation")