import argparse
import pandas as pd
from sklearn.preprocessing import RobustScaler
import os

# -------------------------
# PATH SETUP (FIXED)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

# -------------------------
# ARGUMENT PARSING
# -------------------------
parser = argparse.ArgumentParser(
    description="Scale train/test dataset with RobustScaler"
)

parser.add_argument(
    "dataset_name",
    help="Dataset name, e.g., all_anomalous_cleaned_exp2"
)

args = parser.parse_args()
dataset_name = args.dataset_name

# -------------------------
# FILE PATHS
# -------------------------
TRAIN_FEATURES = os.path.join(TRAIN_DIR, f"X_train_{dataset_name}.csv")
TEST_FEATURES  = os.path.join(TEST_DIR, f"X_test_{dataset_name}.csv")

OUTPUT_TRAIN = os.path.join(TRAIN_DIR, f"X_train_{dataset_name}_scaled.csv")
OUTPUT_TEST  = os.path.join(TEST_DIR, f"X_test_{dataset_name}_scaled.csv")

# -------------------------
# LOAD DATA (SAFE)
# -------------------------
print("Loading datasets...")

if not os.path.exists(TRAIN_FEATURES):
    raise FileNotFoundError(f"Train file not found: {TRAIN_FEATURES}")

if not os.path.exists(TEST_FEATURES):
    raise FileNotFoundError(f"Test file not found: {TEST_FEATURES}")

X_train = pd.read_csv(TRAIN_FEATURES)
X_test  = pd.read_csv(TEST_FEATURES)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# -------------------------
# SCALING
# -------------------------
print("Applying RobustScaler...")

scaler = RobustScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# -------------------------
# SAVE SCALED DATA
# -------------------------
print("Saving scaled datasets...")

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    OUTPUT_TRAIN, index=False
)

pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    OUTPUT_TEST, index=False
)

print("✅ Scaling completed successfully.")
print(f"Scaled train saved to: {OUTPUT_TRAIN}")
print(f"Scaled test saved to: {OUTPUT_TEST}")