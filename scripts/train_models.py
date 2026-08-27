import sys
import os
import json
import time
import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM

# -------------------------------------------------
# Argument
# -------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python train_models.py <dataset_exp_name>")
    sys.exit(1)

exp_name = sys.argv[1]
print(f"\n🚀 Training models for: {exp_name}")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

X_train_path = os.path.join(BASE_DIR, "..", "data", "train", f"X_train_{exp_name}_scaled.csv")
y_train_path = os.path.join(BASE_DIR, "..", "data", "train", f"y_train_{exp_name}.csv")

params_path = os.path.join(BASE_DIR, "..", "results", f"best_params_{exp_name}.json")

models_dir = os.path.join(BASE_DIR, "..", "models", exp_name)
os.makedirs(models_dir, exist_ok=True)

print(f"📁 Saving models to: {models_dir}")

# -------------------------------------------------
# Load data
# -------------------------------------------------
X_train = pd.read_csv(X_train_path).values
y_train = pd.read_csv(y_train_path).values.ravel()

# Normal-only subset (for anomaly models)
X_train_normal = X_train[y_train == 0]

# -------------------------------------------------
# Load best params
# -------------------------------------------------
if not os.path.exists(params_path):
    print("❌ Best params file not found. Run tuning first.")
    sys.exit(1)

with open(params_path, "r") as f:
    best_params = json.load(f)

training_times = {}

# -------------------------------------------------
# KNN
# -------------------------------------------------
print("\n--- Training KNN ---")
knn = KNeighborsClassifier(**best_params["knn"])

start = time.time()
knn.fit(X_train, y_train)
training_times["knn"] = time.time() - start

joblib.dump(knn, os.path.join(models_dir, f"knn_{exp_name}.pkl"))
print(f"⏱️ Time: {training_times['knn']:.2f}s")

# -------------------------------------------------
# Random Forest
# -------------------------------------------------
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(**best_params["random_forest"], random_state=42)

start = time.time()
rf.fit(X_train, y_train)
training_times["rf"] = time.time() - start

joblib.dump(rf, os.path.join(models_dir, f"rf_{exp_name}.pkl"))
print(f"⏱️ Time: {training_times['rf']:.2f}s")

# -------------------------------------------------
# LOF
# -------------------------------------------------
print("\n--- Training LOF ---")
lof = LocalOutlierFactor(**best_params["lof"], novelty=True)

start = time.time()
lof.fit(X_train_normal)
training_times["lof"] = time.time() - start

joblib.dump(lof, os.path.join(models_dir, f"lof_{exp_name}.pkl"))
print(f"⏱️ Time: {training_times['lof']:.2f}s")

# -------------------------------------------------
# Isolation Forest
# -------------------------------------------------
print("\n--- Training Isolation Forest ---")
iso = IsolationForest(**best_params["isolation_forest"], random_state=42)

start = time.time()
iso.fit(X_train_normal)
training_times["if"] = time.time() - start

joblib.dump(iso, os.path.join(models_dir, f"if_{exp_name}.pkl"))
print(f"⏱️ Time: {training_times['if']:.2f}s")

# -------------------------------------------------
# One-Class SVM
# -------------------------------------------------
print("\n--- Training One-Class SVM ---")
ocsvm = OneClassSVM(**best_params["ocsvm"])

start = time.time()
ocsvm.fit(X_train_normal)
training_times["ocsvm"] = time.time() - start

joblib.dump(ocsvm, os.path.join(models_dir, f"ocsvm_{exp_name}.pkl"))
print(f"⏱️ Time: {training_times['ocsvm']:.2f}s")

# -------------------------------------------------
# Save training times
# -------------------------------------------------
times_path = os.path.join(models_dir, "training_times.json")
with open(times_path, "w") as f:
    json.dump(training_times, f, indent=2)

print("\n✅ All models trained and saved successfully!")
print(f"🕒 Training times saved to: {times_path}")