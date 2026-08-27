import sys
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# Argument
# -------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python evaluate_models.py <dataset_exp_name> [X_test_file]")
    sys.exit(1)

exp_name = sys.argv[1]
X_test_file = sys.argv[2] if len(sys.argv) > 2 else None

print(f"\n--- Evaluating models (ROC AUC only) for '{exp_name}' ---")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", exp_name)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")

# -------------------------------------------------
# Check models directory
# -------------------------------------------------
if not os.path.exists(MODELS_DIR):
    print(f"❌ Models directory NOT found: {MODELS_DIR}")
    sys.exit(1)

# -------------------------------------------------
# Test data paths
# -------------------------------------------------
if X_test_file is None:
    X_test_path = os.path.join(DATA_DIR, f"X_test_{exp_name}_scaled.csv")
else:
    X_test_path = os.path.join(DATA_DIR, X_test_file)

y_test_path = os.path.join(DATA_DIR, f"y_test_{exp_name}.csv")

# -------------------------------------------------
# Load data
# -------------------------------------------------
if not os.path.exists(X_test_path):
    print(f"❌ Test file not found: {X_test_path}")
    sys.exit(1)

if not os.path.exists(y_test_path):
    print(f"❌ Label file not found: {y_test_path}")
    sys.exit(1)

X_test = pd.read_csv(X_test_path).values
y_test = pd.read_csv(y_test_path).values.ravel()

print("Test shape:", X_test.shape)

# -------------------------------------------------
# Evaluation function (ROC AUC ONLY)
# -------------------------------------------------
def evaluate_model(name, model, X, y):
    try:
        # Score extraction
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
        else:
            print(f"{name:<20} ❌ No scoring method")
            return

        # Flip anomaly model scores
        if name in ["LOF", "Isolation Forest", "One-Class SVM"]:
            scores = -scores

        auc = roc_auc_score(y, scores)
        print(f"{name:<20} ROC AUC: {auc:.4f}")

    except Exception as e:
        print(f"{name:<20} ❌ Error -> {e}")

# -------------------------------------------------
# Model file mapping
# -------------------------------------------------
models = {
    "k-NN": f"knn_{exp_name}.pkl",
    "Random Forest": f"rf_{exp_name}.pkl",
    "LOF": f"lof_{exp_name}.pkl",
    "Isolation Forest": f"if_{exp_name}.pkl",
    "One-Class SVM": f"ocsvm_{exp_name}.pkl",
}

# -------------------------------------------------
# Evaluate models
# -------------------------------------------------
print("\nModel Performance:\n")

evaluated = False

for name, filename in models.items():
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        print(f"{name:<20} ❌ Missing")
        continue

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"{name:<20} ❌ Load error -> {e}")
        continue

    evaluate_model(name, model, X_test, y_test)
    evaluated = True

# -------------------------------------------------
# Final check
# -------------------------------------------------
if not evaluated:
    print("\n❌ No models evaluated.")
    print("👉 Check model files and paths.")