import argparse
import sys
import pickle
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer

"""
Train models for coffee ratings.

Exercise 1:
  - Train a Linear Regression model to predict `rating` from `100g_USD`
  - Save as `model_1.pickle`

Exercise 2:
  - Train a DecisionTreeRegressor to predict `rating` from `100g_USD` + `roast`
  - Convert categorical `roast` → numeric labels (mapping saved to roast_mapping.json)
  - Save model as `model_2.pickle`

Usage:
    python train.py \
      --data-url https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv \
      --output model_1.pickle
"""

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
DEFAULT_OUTPUT = "model_1.pickle"

REQUIRED_FEATURE = "100g_USD"
TARGET = "rating"
ROAST_COL = "roast"
ROAST_MAPPING_JSON = "roast_mapping.json"
MODEL2_OUTPUT = "model_2.pickle"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Linear Regression (Ex1) and Decision Tree (Ex2) for coffee ratings.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="CSV URL for the coffee dataset.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output pickle filename for Exercise 1 (default: model_1.pickle).")
    parser.add_argument("--model2-output", type=str, default=MODEL2_OUTPUT, help="Output pickle filename for Exercise 2 (default: model_2.pickle).")
    parser.add_argument("--roast-mapping-json", type=str, default=ROAST_MAPPING_JSON, help="Filename to save roast category mapping JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading data from: {args.data_url}")
    try:
        df_full = pd.read_csv(args.data_url)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV from URL: {e}", file=sys.stderr)
        sys.exit(1)

    # -------- Exercise 1: Linear Regression on 100g_USD --------
    # Check required columns for Ex1
    missing_cols = [c for c in [REQUIRED_FEATURE, TARGET] if c not in df_full.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns for Exercise 1: {missing_cols}", file=sys.stderr)
        print(f"[INFO] Available columns: {list(df_full.columns)}", file=sys.stderr)
        sys.exit(2)

    df = df_full[[REQUIRED_FEATURE, TARGET]].copy()

    # Drop rows with missing target; impute feature later
    before = len(df)
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows due to missing target '{TARGET}' (Exercise 1).")

    X = df[[REQUIRED_FEATURE]]
    y = df[TARGET]

    # Build training pipeline: impute numeric feature -> linear regression
    model1 = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])

    print("[INFO] (Ex1) Fitting Linear Regression model...")
    model1.fit(X, y)
    print("[INFO] (Ex1) Training complete.")

    # Save Exercise 1 model
    out_path = args.output
    try:
        with open(out_path, "wb") as f:
            pickle.dump({
                "pipeline": model1,
                "feature_names": [REQUIRED_FEATURE],
                "target_name": TARGET,
                "data_url": args.data_url,
                "exercise": 1,
            }, f)
    except Exception as e:
        print(f"[ERROR] Failed to save Exercise 1 model to {out_path}: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"[INFO] (Ex1) Saved model to: {out_path}")
    print(f"[INFO] (Ex1) Rows used for training: {len(df)}")

    # -------- Exercise 2: Decision Tree on 100g_USD + roast --------
    # Check required columns for Ex2
    ex2_required = [REQUIRED_FEATURE, ROAST_COL, TARGET]
    missing_cols2 = [c for c in ex2_required if c not in df_full.columns]
    if missing_cols2:
        print(f"[ERROR] Missing required columns for Exercise 2: {missing_cols2}", file=sys.stderr)
        sys.exit(4)

    df2 = df_full[ex2_required].copy()

    # Keep rows with target present
    before2 = len(df2)
    df2 = df2.dropna(subset=[TARGET])
    after2 = len(df2)
    if after2 < before2:
        print(f"[WARN] Dropped {before2 - after2} rows due to missing target '{TARGET}' (Exercise 2).")

    # Build roast -> numeric mapping from UNIQUE values present (case-sensitive keep-as-is)
    # You can customize ordering if desired; here we sort for determinism.
    unique_roasts = sorted([r for r in df2[ROAST_COL].dropna().unique().tolist()])
    roast_mapping = {r: i for i, r in enumerate(unique_roasts)}

    # Apply mapping
    df2["roast_numeric"] = df2[ROAST_COL].map(roast_mapping)

    # Preprocessor: impute median for price; most_frequent for roast_numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("price_imputer", SimpleImputer(strategy="median"), [REQUIRED_FEATURE]),
            ("roast_imputer", SimpleImputer(strategy="most_frequent"), ["roast_numeric"]),
        ],
        remainder="drop"
    )

    model2 = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    X2 = df2[[REQUIRED_FEATURE, "roast_numeric"]]
    y2 = df2[TARGET]

    print("[INFO] (Ex2) Fitting DecisionTreeRegressor model...")
    model2.fit(X2, y2)
    print("[INFO] (Ex2) Training complete.")

    # Save roast mapping for later reuse
    try:
        with open(args.roast_mapping_json, "w", encoding="utf-8") as f:
            json.dump(roast_mapping, f, ensure_ascii=False, indent=2)
        print(f"[INFO] (Ex2) Saved roast mapping to: {args.roast_mapping_json}")
    except Exception as e:
        print(f"[ERROR] Failed to save roast mapping JSON: {e}", file=sys.stderr)
        sys.exit(5)

    # Save Exercise 2 model
    try:
        with open(args.model2_output, "wb") as f:
            pickle.dump({
                "pipeline": model2,
                "feature_names": [REQUIRED_FEATURE, "roast_numeric"],
                "original_categorical_feature": ROAST_COL,
                "target_name": TARGET,
                "data_url": args.data_url,
                "roast_mapping_json": args.roast_mapping_json,
                "roast_mapping": roast_mapping,  # also embed a copy
                "exercise": 2,
            }, f)
        print(f"[INFO] (Ex2) Saved model to: {args.model2_output}")
        print(f"[INFO] (Ex2) Rows used for training: {len(df2)}")
    except Exception as e:
        print(f"[ERROR] Failed to save Exercise 2 model to {args.model2_output}: {e}", file=sys.stderr)
        sys.exit(6)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
