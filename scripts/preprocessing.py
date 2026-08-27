import pandas as pd
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_raw_data(filename):
    # If full path is given, use it directly
    if os.path.exists(filename):
        return pd.read_csv(filename)

    # Otherwise look inside RAW_DIR
    file_path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename} or {file_path}")

    return pd.read_csv(file_path)


def preprocess_and_save(input_file, output_file, numeric_only=False):
    df = load_raw_data(input_file)

    print(f"Loaded data shape: {df.shape}")

    if numeric_only:
        df = df.select_dtypes(include=["number"])
        print(f"After numeric_only: {df.shape}")

    # Drop duplicates first (important for ML stability)
    df = df.drop_duplicates()

    # Drop missing values
    df = df.dropna()

    print(f"After cleaning: {df.shape}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Ensure .csv extension
    if not output_file.endswith(".csv"):
        output_file += ".csv"

    save_path = os.path.join(PROCESSED_DIR, output_file)

    df.to_csv(save_path, index=False)

    print(f"✅ Cleaned data saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw dataset")
    parser.add_argument("input_file", help="CSV file name or full path")
    parser.add_argument("-o", "--output_file", required=True, help="Output CSV filename")
    parser.add_argument("--numeric_only", action="store_true", help="Keep only numeric columns")

    args = parser.parse_args()

    preprocess_and_save(
        args.input_file,
        args.output_file,
        numeric_only=args.numeric_only
    )