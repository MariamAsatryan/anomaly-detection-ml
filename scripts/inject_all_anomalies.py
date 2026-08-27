import pandas as pd
import numpy as np
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ANOMALY_DIR = os.path.join(BASE_DIR, "data", "anomalies")


def load_data(filename):
    # Allow full path OR fallback to processed dir
    if os.path.exists(filename):
        return pd.read_csv(filename), os.path.basename(filename)

    file_path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename} or {file_path}")

    return pd.read_csv(file_path), os.path.basename(file_path)


def inject_synthetic_anomalies(filename, contamination=0.05, mode="swap", seed=42):
    np.random.seed(seed)

    df, base_name = load_data(filename)

    print(f"Loaded data shape: {df.shape}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for anomaly injection.")

    n_samples = len(df)
    total_anomalies = int(n_samples * contamination)

    if total_anomalies == 0:
        raise ValueError("Contamination too small, resulted in 0 anomalies.")

    anomaly_idx = np.random.choice(df.index, total_anomalies, replace=False)

    print(f"Injecting {total_anomalies} anomalies using mode='{mode}'")

    if mode == "all":
        thirds = np.array_split(anomaly_idx, 3)
        swap_idx, noise_idx, out_idx = thirds

        # SWAP
        for idx in swap_idx:
            swap_with = np.random.choice(df.index)
            df.loc[idx, numeric_cols] = df.loc[swap_with, numeric_cols].values

        # NOISE
        for idx in noise_idx:
            noise = np.random.normal(0.0, 0.5, size=len(numeric_cols))
            df.loc[idx, numeric_cols] += noise

        # OUTLIERS
        max_vals = df[numeric_cols].max()
        for idx in out_idx:
            extreme_values = np.random.uniform(
                low=max_vals * 1.5,
                high=max_vals * 3.0,
                size=len(numeric_cols)
            )
            df.loc[idx, numeric_cols] = extreme_values

    else:
        for idx in anomaly_idx:
            if mode == "swap":
                swap_with = np.random.choice(df.index)
                df.loc[idx, numeric_cols] = df.loc[swap_with, numeric_cols].values

            elif mode == "noise":
                noise = np.random.normal(0.0, 0.5, size=len(numeric_cols))
                df.loc[idx, numeric_cols] += noise

            elif mode == "outliers":
                max_vals = df[numeric_cols].max()
                extreme_values = np.random.uniform(
                    low=max_vals * 1.5,
                    high=max_vals * 3.0,
                    size=len(numeric_cols)
                )
                df.loc[idx, numeric_cols] = extreme_values

    # Add label column
    df["Class"] = 0
    df.loc[anomaly_idx, "Class"] = 1

    # Ensure output directory exists
    os.makedirs(ANOMALY_DIR, exist_ok=True)

    # 🔥 FIXED NAMING (critical for pipeline consistency)
    clean_name = base_name.replace(".csv", "")
    outname = f"{mode}_anomalous_{clean_name}.csv"

    save_path = os.path.join(ANOMALY_DIR, outname)
    df.to_csv(save_path, index=False)

    print(f"Final data shape: {df.shape}")
    print(f"✅ Anomalous data saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject synthetic anomalies into dataset")

    parser.add_argument("input_file", help="Cleaned CSV file")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("-m", "--mode",
                        choices=["swap", "noise", "outliers", "all"],
                        default="swap")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    inject_synthetic_anomalies(
        args.input_file,
        contamination=args.contamination,
        mode=args.mode,
        seed=args.seed
    )