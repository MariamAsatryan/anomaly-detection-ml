import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, make_scorer

warnings.filterwarnings("ignore")

# -------------------------------------------------
# Argument
# -------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python tune_models.py <dataset_exp_name>")
    sys.exit(1)

exp_name = sys.argv[1]
print(f"\n🔍 Tuning models for: {exp_name}")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
X_train_path = os.path.join(BASE_DIR, "..", "data", "train", f"X_train_{exp_name}_scaled.csv")
y_train_path = os.path.join(BASE_DIR, "..", "data", "train", f"y_train_{exp_name}.csv")

# -------------------------------------------------
# Load data
# -------------------------------------------------
X = pd.read_csv(X_train_path).values
y = pd.read_csv(y_train_path).values.ravel()

print("Train shape:", X.shape)
print("Normal samples:", np.sum(y == 0))

# Normal-only training set
X_normal = X[y == 0]

results = {}

# -------------------------------------------------
# Custom scorer (robust)
# -------------------------------------------------
def anomaly_scorer(estimator, X_val, y_val):
    try:
        scores = estimator.decision_function(X_val)
    except:
        scores = estimator.score_samples(X_val)

    # IMPORTANT: anomalies should be higher score
    scores = -scores

    # If only one class → return 0 (avoid nan crash)
    if len(np.unique(y_val)) < 2:
        return 0.0

    return roc_auc_score(y_val, scores)

scorer = make_scorer(anomaly_scorer, needs_proba=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------------------------
# KNN
# -------------------------------------------------
print("\n--- KNN ---")

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

knn_grid.fit(X, y)

print("Best KNN:", knn_grid.best_params_)
print("ROC AUC:", knn_grid.best_score_)

results["knn"] = knn_grid.best_params_

# -------------------------------------------------
# Random Forest
# -------------------------------------------------
print("\n--- Random Forest ---")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        "n_estimators": [50, 100],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5]
    },
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

rf_grid.fit(X, y)

print("Best RF:", rf_grid.best_params_)
print("ROC AUC:", rf_grid.best_score_)

results["random_forest"] = rf_grid.best_params_

# -------------------------------------------------
# LOF (FIXED PROPERLY)
# -------------------------------------------------
print("\n--- LOF ---")

lof_results = []
best_score = -1
best_params = None

for n in [10, 20]:
    for metric in ["euclidean", "manhattan"]:
        for contamination in [0.01, 0.05]:

            model = LocalOutlierFactor(
                n_neighbors=n,
                metric=metric,
                contamination=contamination,
                novelty=True
            )

            model.fit(X_normal)

            scores = -model.decision_function(X)

            if len(np.unique(y)) < 2:
                auc = 0
            else:
                auc = roc_auc_score(y, scores)

            if auc > best_score:
                best_score = auc
                best_params = {
                    "n_neighbors": n,
                    "metric": metric,
                    "contamination": contamination
                }

print("Best LOF:", best_params)
print("ROC AUC:", best_score)

results["lof"] = best_params

# -------------------------------------------------
# Isolation Forest
# -------------------------------------------------
print("\n--- Isolation Forest ---")

best_score = -1
best_params = None

for n in [50, 100]:
    for ms in ["auto", 0.8]:
        for c in [0.01, 0.05]:

            model = IsolationForest(
                n_estimators=n,
                max_samples=ms,
                contamination=c,
                random_state=42
            )

            model.fit(X_normal)

            scores = -model.decision_function(X)

            auc = roc_auc_score(y, scores)

            if auc > best_score:
                best_score = auc
                best_params = {
                    "n_estimators": n,
                    "max_samples": ms,
                    "contamination": c
                }

print("Best IF:", best_params)
print("ROC AUC:", best_score)

results["isolation_forest"] = best_params

# -------------------------------------------------
# OC-SVM
# -------------------------------------------------
print("\n--- OC-SVM ---")

best_score = -1
best_params = None

for gamma in ["scale", "auto"]:
    for nu in [0.01, 0.05]:

        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
        model.fit(X_normal)

        scores = -model.decision_function(X)

        auc = roc_auc_score(y, scores)

        if auc > best_score:
            best_score = auc
            best_params = {
                "kernel": "rbf",
                "gamma": gamma,
                "nu": nu
            }

print("Best OC-SVM:", best_params)
print("ROC AUC:", best_score)

results["ocsvm"] = best_params

# -------------------------------------------------
# Save
# -------------------------------------------------
results_dir = os.path.join(BASE_DIR, "..", "results")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, f"best_params_{exp_name}.json")

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved best params to: {output_path}")