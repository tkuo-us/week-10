import argparse
import sys
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np

"""
Exercise 1:
  Train a Linear Regression model to predict `rating` from `100g_USD`
  and save it as `model_1.pickle` (the saved object IS a sklearn LinearRegression).

Exercise 2:
  Train a DecisionTreeRegressor to predict `rating` from `100g_USD` + `roast`
  (categorical -> numeric via OrdinalEncoder, with imputation done BEFORE fitting)
  and save it as `model_2.pickle` (the saved object IS a sklearn DecisionTreeRegressor).

Usage:
    python train.py \
      --data-url https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv \
      --output model_1.pickle \
      --model2-output model_2.pickle
"""

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
DEFAULT_OUTPUT = "model_1.pickle"
DEFAULT_MODEL2_OUTPUT = "model_2.pickle"

PRICE_COL = "100g_USD"
ROAST_COL = "roast"
TARGET = "rating"


def parse_args():
    p = argparse.ArgumentParser(description="Train Exercise 1 & 2 models for coffee ratings.")
    p.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="CSV URL for the coffee dataset.")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output filename for Exercise 1 (LinearRegression).")
    p.add_argument("--model2-output", type=str, default=DEFAULT_MODEL2_OUTPUT, help="Output filename for Exercise 2 (DecisionTreeRegressor).")
    return p.parse_args()


def load_csv_or_die(url: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {url}")
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}", file=sys.stderr)
        sys.exit(1)


def exercise1_train_and_save(df: pd.DataFrame, out_path: str):
    # ----- checks -----
    for col in [PRICE_COL, TARGET]:
        if col not in df.columns:
            print(f"[ERROR] (Ex1) Missing required column: {col}", file=sys.stderr)
            print(f"[INFO] Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)

    # ----- select & clean -----
    df1 = df[[PRICE_COL, TARGET]].copy()

    # check target validity
    df1[TARGET] = pd.to_numeric(df1[TARGET], errors="coerce")
    before = len(df1)
    df1 = df1.dropna(subset=[TARGET])
    after = len(df1)
    if after < before:
        print(f"[WARN] (Ex1) Dropped {before - after} rows with missing/invalid '{TARGET}'.")

    # median impute price
    X = df1[[PRICE_COL]]
    y = df1[TARGET]
    price_imputer = SimpleImputer(strategy="median")
    X_imputed = price_imputer.fit_transform(X)

    # ----- fit LR -----
    model = LinearRegression()
    print("[INFO] (Ex1) Fitting LinearRegression...")
    model.fit(X_imputed, y)
    print("[INFO] (Ex1) Done. Coef=%.6f, Intercept=%.6f" % (model.coef_[0], model.intercept_))

    # ----- save pure sklearn model -----
    try:
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[INFO] (Ex1) Saved LinearRegression to: {out_path}")
    except Exception as e:
        print(f"[ERROR] (Ex1) Failed to save model to {out_path}: {e}", file=sys.stderr)
        sys.exit(3)


def exercise2_train_and_save(df: pd.DataFrame, out_path: str):
    # ----- checks -----
    for col in [PRICE_COL, ROAST_COL, TARGET]:
        if col not in df.columns:
            print(f"[ERROR] (Ex2) Missing required column: {col}", file=sys.stderr)
            print(f"[INFO] Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(4)

    # ----- select & clean -----
    df2 = df[[PRICE_COL, ROAST_COL, TARGET]].copy()

    # check target validity
    df2[TARGET] = pd.to_numeric(df2[TARGET], errors="coerce")
    before = len(df2)
    df2 = df2.dropna(subset=[TARGET])
    after = len(df2)
    if after < before:
        print(f"[WARN] (Ex2) Dropped {before - after} rows with missing/invalid '{TARGET}'.")

    # median impute 100g_USD
    price_imputer = SimpleImputer(strategy="median")
    price_vals = price_imputer.fit_transform(df2[[PRICE_COL]])

    # roast -> string & fillna
    roast_series = df2[ROAST_COL].astype(str)
    roast_series = roast_series.replace(["nan", "NaN", "None"], np.nan)
    roast_series = roast_series.fillna("Unknown")

    # Ordinal encode roast
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    roast_num = enc.fit_transform(roast_series.to_frame())

    # final X, y
    X2 = np.hstack([price_vals, roast_num])
    y2 = df2[TARGET].values

    # ----- fit Decision Tree -----
    model = DecisionTreeRegressor(random_state=42)
    print("[INFO] (Ex2) Fitting DecisionTreeRegressor...")
    model.fit(X2, y2)
    print("[INFO] (Ex2) Done. Tree depth=%s" % getattr(model, "get_depth", lambda: "N/A")())

    # ----- save pure sklearn model -----
    try:
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[INFO] (Ex2) Saved DecisionTreeRegressor to: {out_path}")
    except Exception as e:
        print(f"[ERROR] (Ex2) Failed to save model to {out_path}: {e}", file=sys.stderr)
        sys.exit(5)


def main():
    args = parse_args()
    df = load_csv_or_die(args.data_url)

    # Exercise 1
    exercise1_train_and_save(df, args.output)

    # Exercise 2
    exercise2_train_and_save(df, args.model2_output)

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
