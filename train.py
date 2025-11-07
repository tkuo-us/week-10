import argparse
import sys
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

"""
Train a Linear Regression model to predict `rating` from `100g_USD`
and save it as `model_1.pickle`.

Usage:
    python train.py \
      --data-url https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv \
      --output model_1.pickle
"""

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
DEFAULT_OUTPUT = "model_1.pickle"

REQUIRED_FEATURE = "100g_USD"
TARGET = "rating"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Linear Regression for coffee ratings.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="CSV URL for the coffee dataset.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output pickle filename.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading data from: {args.data_url}")
    try:
        df = pd.read_csv(args.data_url)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV from URL: {e}", file=sys.stderr)
        sys.exit(1)

    # Check required columns
    missing_cols = [c for c in [REQUIRED_FEATURE, TARGET] if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}", file=sys.stderr)
        print(f"[INFO] Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    # Keep only necessary columns
    df = df[[REQUIRED_FEATURE, TARGET]].copy()

    # Drop rows with missing target; impute feature later
    before = len(df)
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows due to missing target '{TARGET}'.")

    X = df[[REQUIRED_FEATURE]]
    y = df[TARGET]

    # Build training pipeline: impute numeric feature -> linear regression
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])

    print("[INFO] Fitting Linear Regression model...")
    model.fit(X, y)
    print("[INFO] Training complete.")

    # Save to pickle
    out_path = args.output
    try:
        with open(out_path, "wb") as f:
            pickle.dump({
                "pipeline": model,
                "feature_names": [REQUIRED_FEATURE],
                "target_name": TARGET,
                "data_url": args.data_url,
            }, f)
    except Exception as e:
        print(f"[ERROR] Failed to save model to {out_path}: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"[INFO] Saved model to: {out_path}")
    print(f"[INFO] Rows used for training: {len(df)}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
