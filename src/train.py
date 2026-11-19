"""Train the compensation model.

Question: given only what you know about a role before anyone is paid
(job title, job family, department, organisation group, union), what total
compensation should you expect in a given year?

Nothing about the individual's actual pay goes in. The six pay components
add up exactly to total compensation, so using them would be leakage.

Approach
--------
* Population: full-year records from complete fiscal years (see clean.py).
* Pay level vs. pay structure: city-wide pay rises every year, and tree
  models cannot extrapolate. So we split the problem in two:
    1. a city-wide pay index (median total comp per year), projected forward
       with a log-linear trend;
    2. a LightGBM model of each role's pay *relative* to that index.
  prediction = projected index x exp(relative pay)
* Validation is by time: train on earlier years, test on the latest complete
  year. No employee-year from the test year is ever seen in training.
* Quantile models (10th / 90th percentile) give an 80% pay range.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "compensation.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CAT_FEATURES = ["job_code", "job_family", "department_code", "organization_group", "union_code"]
FEATURES = CAT_FEATURES + ["year"]
FEATURE_LABELS = {
    "job_code": "Job title", "job_family": "Job family", "department_code": "Department",
    "organization_group": "Organisation group", "union_code": "Union", "year": "Year",
}
TARGET = "total_compensation"

BASE_PARAMS = dict(
    learning_rate=0.1, num_leaves=31, min_data_in_leaf=40,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
    cat_smooth=10, cat_l2=10, max_cat_threshold=64,
    seed=42, verbose=-1, num_threads=4,
)


# ---------------------------------------------------------------- data
def load_model_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df = df[df["is_full_year"] & df["is_complete_year"]].copy()
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).astype("category")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- pay index
def fit_index(df: pd.DataFrame) -> dict:
    """Median total comp per year + log-linear trend for projecting forward."""
    med = df.groupby("year")[TARGET].median()
    slope, intercept = np.polyfit(med.index.values, np.log(med.values), 1)
    return {"medians": {int(k): float(v) for k, v in med.items()},
            "slope": float(slope), "intercept": float(intercept)}


def index_for(years, idx: dict) -> np.ndarray:
    """Observed median where we have it, trend projection otherwise."""
    years = np.asarray(years)
    trend = np.exp(idx["intercept"] + idx["slope"] * years)
    obs = np.array([idx["medians"].get(int(y), np.nan) for y in years])
    return np.where(np.isnan(obs), trend, obs)


# ---------------------------------------------------------------- model
def _dataset(df, idx, ref=None):
    y = np.log(df[TARGET].values) - np.log(index_for(df["year"], idx))
    return lgb.Dataset(df[FEATURES], y, categorical_feature=CAT_FEATURES,
                       reference=ref, free_raw_data=False)


def fit_models(train: pd.DataFrame, idx: dict, n_rounds: int) -> dict:
    ds = _dataset(train, idx)
    models = {}
    for name, extra in {
        "p50": dict(objective="regression_l1"),
        "p10": dict(objective="quantile", alpha=0.10),
        "p90": dict(objective="quantile", alpha=0.90),
    }.items():
        models[name] = lgb.train({**BASE_PARAMS, **extra}, ds, num_boost_round=n_rounds)
    return models


def find_rounds(df: pd.DataFrame, valid_year: int) -> int:
    """Early stopping on the year before the test year."""
    tr, va = df[df["year"] < valid_year], df[df["year"] == valid_year]
    idx = fit_index(tr)
    dtr = _dataset(tr, idx)
    # the validation year uses the projected index, exactly like the real test
    dva = lgb.Dataset(va[FEATURES],
                      np.log(va[TARGET].values) - np.log(index_for(va["year"], idx)),
                      categorical_feature=CAT_FEATURES, reference=dtr)
    booster = lgb.train({**BASE_PARAMS, "objective": "regression_l1"}, dtr,
                        num_boost_round=1000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster.best_iteration


def predict(models: dict, X: pd.DataFrame, idx: dict, widen: float = 1.0) -> pd.DataFrame:
    """Point prediction plus an 80% range.

    `widen` stretches the quantile band (in log space) so that it really
    covers ~80% of people in an unseen year. It is calibrated on the
    validation year in `calibrate_band`.
    """
    level = index_for(X["year"], idx)
    raw = {k: m.predict(X[FEATURES]) for k, m in models.items()}
    mid = raw["p50"]
    lo = mid - widen * np.maximum(mid - raw["p10"], 0)
    hi = mid + widen * np.maximum(raw["p90"] - mid, 0)
    return pd.DataFrame({"p10": level * np.exp(lo), "p50": level * np.exp(mid),
                         "p90": level * np.exp(hi)}, index=X.index)


def calibrate_band(df: pd.DataFrame, valid_year: int, n_rounds: int, target: float = 0.80) -> float:
    """Smallest stretch factor that gives `target` coverage on the validation year."""
    tr, va = df[df["year"] < valid_year], df[df["year"] == valid_year]
    idx = fit_index(tr)
    models = fit_models(tr, idx, n_rounds)
    y = va[TARGET].values
    for widen in np.arange(1.0, 3.01, 0.05):
        p = predict(models, va, idx, widen)
        if ((y >= p["p10"]) & (y <= p["p90"])).mean() >= target:
            return float(round(widen, 2))
    return 3.0


# ---------------------------------------------------------------- metrics
def metrics(y, p, lo=None, hi=None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    m = {
        "R2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "MAE": float(np.abs(y - p).mean()),
        "MedAPE": float(np.median(np.abs(y - p) / y)),
        "within_10pct": float((np.abs(y - p) / y <= 0.10).mean()),
    }
    if lo is not None:
        m["coverage_80"] = float(((y >= lo) & (y <= hi)).mean())
    return m


def baselines(train: pd.DataFrame, test: pd.DataFrame, idx: dict) -> dict:
    growth = index_for(test["year"], idx) / idx["medians"][int(train["year"].max())]
    city = np.full(len(test), index_for(test["year"], idx))
    last = train[train["year"] == train["year"].max()]
    job_med = last.groupby("job_code", observed=True)[TARGET].median()
    job_all = train.groupby("job_code", observed=True)[TARGET].median()
    job_pred = test["job_code"].astype(str).map(job_med).fillna(
        test["job_code"].astype(str).map(job_all)).astype(float)
    job_pred = job_pred.fillna(float(np.median(train[TARGET]))) * growth
    return {
        "City-wide median": metrics(test[TARGET], city),
        "Job title median (last year, grown)": metrics(test[TARGET], job_pred),
    }


def importance(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    contrib = models["p50"].predict(X[FEATURES], pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    return (pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                          "mean_abs_log_effect": imp})
            .sort_values("mean_abs_log_effect", ascending=False)
            .assign(share=lambda d: d["mean_abs_log_effect"] / d["mean_abs_log_effect"].sum()))


# ---------------------------------------------------------------- main
def main() -> None:
    df = load_model_data()
    years = sorted(df["year"].unique())
    test_year, valid_year = years[-1], years[-2]
    print(f"rows {len(df):,} | years {years[0]}-{test_year} | test year {test_year}")

    n_rounds = find_rounds(df[df["year"] < test_year], valid_year)
    print(f"boosting rounds (early stopping on {valid_year}): {n_rounds}")
    widen = calibrate_band(df[df["year"] < test_year], valid_year, n_rounds)
    print(f"band stretch factor (calibrated on {valid_year}): {widen}")

    # ---- honest evaluation: train < test_year, test = test_year
    train, test = df[df["year"] < test_year], df[df["year"] == test_year]
    idx = fit_index(train)
    models = fit_models(train, idx, n_rounds)
    pred = predict(models, test, idx, widen)

    results = baselines(train, test, idx)
    results["LightGBM (this model)"] = metrics(test[TARGET], pred["p50"], pred["p10"], pred["p90"])
    comp = pd.DataFrame(results).T
    print(comp.round(3).to_string())

    REPORT_DIR.mkdir(exist_ok=True)
    comp.to_csv(REPORT_DIR / "model_comparison.csv")
    imp = importance(models, test.sample(min(20000, len(test)), random_state=42))
    imp.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(imp.round(3).to_string(index=False))

    # error by job family on the test year
    err = (test.assign(pred=pred["p50"], ape=lambda d: (d["pred"] - d[TARGET]).abs() / d[TARGET])
           .groupby("job_family", observed=True)
           .agg(n=("ape", "size"), MedAPE=("ape", "median"))
           .query("n >= 100").sort_values("MedAPE"))
    err.to_csv(REPORT_DIR / "error_by_job_family.csv")

    # ---- production model: refit on every complete year
    idx_full = fit_index(df)
    final = fit_models(df, idx_full, n_rounds)
    MODEL_DIR.mkdir(exist_ok=True)
    for k, m in final.items():
        m.save_model(str(MODEL_DIR / f"lgbm_{k}.txt"))

    meta = {
        "features": FEATURES, "categorical_features": CAT_FEATURES,
        "train_years": [int(years[0]), int(test_year)],
        "eval": {"train_years": [int(years[0]), int(valid_year)], "test_year": int(test_year),
                 "n_train": int(len(train)), "n_test": int(len(test)),
                 "results": results},
        "n_rounds": int(n_rounds),
        "band_widen": widen,
        "pay_index": idx_full,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("saved models/ and reports/")


if __name__ == "__main__":
    main()
