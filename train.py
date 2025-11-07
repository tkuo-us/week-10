import argparse
import sys
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

"""
Exercise 1:
  Train a Linear Regression model to predict `rating` from `100g_USD`
  and save it as `model_1.pickle` (the saved object IS a sklearn model/pipeline).

Exercise 2:
  Train a DecisionTreeRegressor to predict `rating` from `100g_USD` + `roast`
  (categorical -> numeric via OrdinalEncoder) and save it as `model_2.pickle`
  (also a sklearn pipeline with a `.predict` method).
"""

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
DEFAULT_OUTPUT = "model_1.pickle"
MODEL2_OUTPUT = "model_2.pickle"

REQUIRED_FEATURE = "100g_USD"
TARGET = "rating"
ROAST_COL = "roast"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Exercise 1 and 2 models for coffee ratings.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="CSV URL for the coffee dataset.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output pickle filename for Exercise 1 (default: model_1.pickle).")
    parser.add_argument("--model2-output", type=str, default=MODEL2_OUTPUT, help="Output pickle filename for Exercise 2 (default: model_2.pickle).")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading data from: {args.data_url}")
    try:
        df_full = pd.read_csv(args.data_url)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV from URL: {e}", file=sys.stderr)
        sys.exit(1)

    # ---------- Exercise 1: Linear Regression on 100g_USD ----------
    for col in [REQUIRED_FEATURE, TARGET]:
        if col not in df_full.columns:
            print(f"[ERROR] Missing required column for Exercise 1: {col}", file=sys.stderr)
            print(f"[INFO] Available columns: {list(df_full.columns)}", file=sys.stderr)
            sys.exit(2)

    df1 = df_full[[REQUIRED_FEATURE, TARGET]].copy()
    before = len(df1)
    df1 = df1.dropna(subset=[TARGET])  # drop rows with missing target
    after = len(df1)
    if after < before:
        print(f"[WARN] (Ex1) Dropped {before - after} rows due to missing target '{TARGET}'.")

    X1 = df1[[REQUIRED_FEATURE]]
    y1 = df1[TARGET]

    model1 = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])

    print("[INFO] (Ex1) Fitting Linear Regression model...")
    model1.fit(X1, y1)
    print("[INFO] (Ex1) Training complete.")

    # IMPORTANT: Save the sklearn model/pipeline object directly
    try:
        with open(args.output, "wb") as f:
            pickle.dump(model1, f)
        print(f"[INFO] (Ex1) Saved sklearn model to: {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to save Exercise 1 model to {args.output}: {e}", file=sys.stderr)
        sys.exit(3)

    # ---------- Exercise 2: Decision Tree on 100g_USD + roast ----------
    for col in [REQUIRED_FEATURE, ROAST_COL, TARGET]:
        if col not in df_full.columns:
            print(f"[ERROR] Missing required column for Exercise 2: {col}", file=sys.stderr)
            sys.exit(4)

    df2 = df_full[[REQUIRED_FEATURE, ROAST_COL, TARGET]].copy()
    before2 = len(df2)
    df2 = df2.dropna(subset=[TARGET])  # target must be present
    after2 = len(df2)
    if after2 < before2:
        print(f"[WARN] (Ex2) Dropped {before2 - after2} rows due to missing target '{TARGET}'.")

    X2 = df2[[REQUIRED_FEATURE, ROAST_COL]]
    y2 = df2[TARGET]

    # Preprocess:
    # - 100g_USD: impute median
    # - roast: impute most_frequent, then OrdinalEncode to numeric labels
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), [REQUIRED_FEATURE]),
            ("cat", cat_pipe, [ROAST_COL]),
        ],
        remainder="drop"
    )

    model2 = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    print("[INFO] (Ex2) Fitting DecisionTreeRegressor model...")
    model2.fit(X2, y2)
    print("[INFO] (Ex2) Training complete.")

    # IMPORTANT: Save the sklearn model/pipeline object directly
    try:
        with open(args.model2_output, "wb") as f:
            pickle.dump(model2, f)
        print(f"[INFO] (Ex2) Saved sklearn model to: {args.model2_output}")
    except Exception as e:
        print(f"[ERROR] Failed to save Exercise 2 model to {args.model2_output}: {e}", file=sys.stderr)
        sys.exit(5)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
