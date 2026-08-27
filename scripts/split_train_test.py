import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ANOMALY_DIR = os.path.join(BASE_DIR, "data", "anomalies")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def load_data(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename), os.path.basename(filename)

    file_path = os.path.join(ANOMALY_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename} or {file_path}")

    return pd.read_csv(file_path), os.path.basename(file_path)


def split_and_save(filename, test_size=0.2, random_state=42):
    df, base_name = load_data(filename)

    print(f"Loaded data shape: {df.shape}")

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    # Separate features and labels
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 🔥 FIX: remove .csv extension for consistency across pipeline
    clean_name = base_name.replace(".csv", "")

    train_X_path = os.path.join(TRAIN_DIR, f"X_train_{clean_name}.csv")
    train_y_path = os.path.join(TRAIN_DIR, f"y_train_{clean_name}.csv")
    test_X_path = os.path.join(TEST_DIR, f"X_test_{clean_name}.csv")
    test_y_path = os.path.join(TEST_DIR, f"y_test_{clean_name}.csv")

    # Save
    X_train.to_csv(train_X_path, index=False)
    y_train.to_csv(train_y_path, index=False)
    X_test.to_csv(test_X_path, index=False)
    y_test.to_csv(test_y_path, index=False)

    print("\n✅ Train/test split completed.")
    print(f"Train X: {train_X_path}")
    print(f"Train y: {train_y_path}")
    print(f"Test X:  {test_X_path}")
    print(f"Test y:  {test_y_path}")

    print("\n📊 Anomaly statistics:")
    print(f"Total anomalies: {y.sum()} ({100 * y.mean():.2f}%)")
    print(f"Train anomalies: {y_train.sum()} ({100 * y_train.mean():.2f}%)")
    print(f"Test anomalies:  {y_test.sum()} ({100 * y_test.mean():.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split anomalous dataset into train/test sets")
    parser.add_argument("input_file", help="Anomalous CSV file")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    split_and_save(
        args.input_file,
        test_size=args.test_size,
        random_state=args.random_state
    )