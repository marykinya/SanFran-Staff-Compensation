"""Clean the raw DataSF Employee Compensation export.

Input : data/Employee_Compensation.csv  (export of https://data.sfgov.org/d/88g8-5mnd)
Output: data/processed/compensation.parquet

Key decisions
-------------
1. Fiscal years only. The raw export lists every employee twice per year
   (once for the calendar year, once for the fiscal year). Mixing both
   double counts people and leaks between train and test.
2. Codes are the keys, not names. Departments, jobs and unions were renamed
   over time ("DPH Public Health" -> "Public Health"). Each code gets its
   most recent name, so one department is one category.
3. Real zeros stay zero. Most employees have no overtime, and that is a
   meaningful fact, not a missing value.
4. Incomplete years are flagged. In the Feb 2023 export, health & dental is
   zero for ~93% of FY2022 records, so FY2022 totals are understated. Any
   year where most full-year records report $0 health & dental is marked
   incomplete and left out of modelling and trend charts.
5. Part-year and part-time records are flagged, not deleted. We cannot see
   hours worked, so a record counts as "full year" when base salary is at
   least half the median base salary for that job code in that year.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "Employee_Compensation.csv"
OUT_PATH = ROOT / "data" / "processed" / "compensation.parquet"

COMPONENTS = [
    "salaries", "overtime", "other_salaries",
    "retirement", "health_and_dental", "other_benefits",
]
FULL_YEAR_SHARE = 0.5      # base salary >= 50% of job-year median
FULL_YEAR_FLOOR = 15_000   # and at least $15k of base salary


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _latest_label(df: pd.DataFrame, code: str, label: str) -> pd.Series:
    """Map each code to the label it carried in the most recent year."""
    latest = (
        df.dropna(subset=[label])
        .sort_values("year")
        .groupby(code)[label]
        .last()
    )
    return df[code].map(latest)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [_snake(c) for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["year_type"].str.lower() == "fiscal"].copy()

    # Codes as strings so "0111" keeps its leading zero
    for c in ["job_code", "job_family_code", "department_code", "union_code"]:
        df[c] = df[c].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    df["union_code"] = df["union_code"].fillna("NONE")
    df["union"] = df["union"].fillna("Unrepresented / unknown")

    df = df.dropna(subset=["department_code", "job_code"])

    # One name per code (latest wins)
    df["department"] = (
        _latest_label(df, "department_code", "department")
        .str.replace(r"^[A-Z]{2,4} (?=[A-Z])", "", regex=True)  # "WTR Water Enterprise" -> "Water Enterprise"
    )
    df["job"] = _latest_label(df, "job_code", "job")
    df["job_family"] = _latest_label(df, "job_code", "job_family")
    df["union"] = _latest_label(df, "union_code", "union")
    df["organization_group"] = _latest_label(df, "department_code", "organization_group")

    # Recompute total from components so the identity always holds
    df["total_compensation"] = df[COMPONENTS].sum(axis=1)
    df = df[df["total_compensation"] > 0]

    # Full-year proxy
    job_year_median = df.groupby(["job_code", "year"])["salaries"].transform("median")
    df["is_full_year"] = (
        (df["salaries"] >= FULL_YEAR_SHARE * job_year_median)
        & (df["salaries"] >= FULL_YEAR_FLOOR)
    )

    # Incomplete-year check (benefits not yet posted)
    hd_missing = (
        df[df["is_full_year"]]
        .assign(hd_zero=lambda d: d["health_and_dental"] == 0)
        .groupby("year")["hd_zero"].mean()
    )
    incomplete_years = hd_missing[hd_missing > 0.5].index.tolist()
    df["is_complete_year"] = ~df["year"].isin(incomplete_years)

    keep = [
        "year", "employee_identifier",
        "organization_group", "department_code", "department",
        "job_family", "job_code", "job", "union_code", "union",
        *COMPONENTS, "total_compensation", "is_full_year", "is_complete_year",
    ]
    df = df[keep].reset_index(drop=True)
    for c in ["organization_group", "department_code", "department",
              "job_family", "job_code", "job", "union_code", "union"]:
        df[c] = df[c].astype("category")
    return df


def main() -> None:
    raw = load_raw()
    df = clean(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"raw rows            {len(raw):>9,}")
    print(f"fiscal rows kept    {len(df):>9,}")
    print(f"full-year rows      {int(df['is_full_year'].sum()):>9,}")
    print(f"years               {df['year'].min()}-{df['year'].max()}")
    bad = sorted(df.loc[~df["is_complete_year"], "year"].unique().tolist())
    print(f"incomplete years    {bad if bad else 'none'}")
    print(f"departments         {df['department_code'].nunique():>9}")
    print(f"job titles          {df['job_code'].nunique():>9}")
    print(f"unions              {df['union_code'].nunique():>9}")
    print(f"saved -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
