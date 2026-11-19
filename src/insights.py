"""Build the small summary tables behind the app and the README.

The app never loads the 400k-row dataset. It reads these CSVs (a few
hundred KB in total), which keeps the deployed app fast.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "compensation.parquet"
OUT = ROOT / "app_data"

SALARY = ["salaries", "overtime", "other_salaries"]
BENEFITS = ["retirement", "health_and_dental", "other_benefits"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    for c in df.select_dtypes("category"):
        df[c] = df[c].astype(str)
    df["benefits"] = df[BENEFITS].sum(axis=1)
    df["cash_pay"] = df[SALARY].sum(axis=1)
    return df


def pay_trend(df: pd.DataFrame) -> pd.DataFrame:
    fy = df[df["is_full_year"] & df["is_complete_year"]]
    t = fy.groupby("year").agg(
        employees=("total_compensation", "size"),
        median_total=("total_compensation", "median"),
        median_base=("salaries", "median"),
        median_benefits=("benefits", "median"),
        overtime_total=("overtime", "sum"),
        cash_total=("cash_pay", "sum"),
        share_with_overtime=("overtime", lambda s: (s > 0).mean()),
    )
    t["overtime_share_of_cash_pay"] = t.pop("overtime_total") / t.pop("cash_total")
    base_year = t.index.min()
    for c in ["median_total", "median_base", "median_benefits"]:
        t[c.replace("median_", "index_")] = 100 * t[c] / t.loc[base_year, c]
    t["headcount_all_records"] = df[df["is_complete_year"]].groupby("year").size()
    return t.reset_index()


def overtime_by_department(df: pd.DataFrame, year: int) -> pd.DataFrame:
    fy = df[df["is_full_year"] & (df["year"] == year)]
    g = fy.groupby("department").agg(
        employees=("overtime", "size"),
        overtime_total=("overtime", "sum"),
        cash_total=("cash_pay", "sum"),
    )
    g["share_ot_over_25pct_of_base"] = (
        (fy["overtime"] >= 0.25 * fy["salaries"]).groupby(fy["department"]).mean()
    )
    g["overtime_share_of_cash_pay"] = g["overtime_total"] / g["cash_total"]
    g["overtime_total_musd"] = g["overtime_total"] / 1e6
    g = g.drop(columns=["cash_total", "overtime_total"])
    return g[g["employees"] >= 50].sort_values("overtime_share_of_cash_pay", ascending=False).reset_index()


def same_job_gaps(df: pd.DataFrame, year: int, min_n: int = 15) -> pd.DataFrame:
    """Same job title, different department: how much does pay differ?"""
    fy = df[df["is_full_year"] & (df["year"] == year)]
    g = (fy.groupby(["job_code", "job", "department"])
         .agg(n=("total_compensation", "size"),
              median_total=("total_compensation", "median"),
              median_base=("salaries", "median"),
              median_overtime=("overtime", "median"))
         .reset_index())
    g = g[g["n"] >= min_n]
    multi = g.groupby("job_code")["department"].transform("nunique") >= 2
    g = g[multi]
    summary = (g.groupby(["job_code", "job"])
               .agg(departments=("department", "nunique"),
                    employees=("n", "sum"),
                    low_total=("median_total", "min"),
                    high_total=("median_total", "max"),
                    low_base=("median_base", "min"),
                    high_base=("median_base", "max"))
               .reset_index())
    summary["total_gap"] = summary["high_total"] - summary["low_total"]
    summary["total_gap_pct"] = summary["total_gap"] / summary["low_total"]
    summary["base_gap_pct"] = (summary["high_base"] - summary["low_base"]) / summary["low_base"]
    return summary.sort_values("total_gap", ascending=False).reset_index(drop=True), g


def job_lookup(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Pay distribution per job title in the latest complete year (fallback: last 3 years)."""
    fy = df[df["is_full_year"] & df["is_complete_year"]]
    recent = fy[fy["year"] >= year - 2]
    def q(p):
        return lambda s: s.quantile(p)

    def summarise(d):
        return d.groupby("job_code").agg(
            job=("job", "first"), job_family=("job_family", "first"),
            n=("total_compensation", "size"),
            p10=("total_compensation", q(.10)), p25=("total_compensation", q(.25)),
            p50=("total_compensation", "median"), p75=("total_compensation", q(.75)),
            p90=("total_compensation", q(.90)),
            median_base=("salaries", "median"), median_overtime=("overtime", "median"),
            median_benefits=("benefits", "median"))
    latest = summarise(recent[recent["year"] == year]).assign(source_years=str(year))
    fallback = summarise(recent).assign(source_years=f"{year-2}-{year}")
    fallback = fallback[~fallback.index.isin(latest.index)]
    return pd.concat([latest, fallback]).reset_index()


def combos(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Job / department / union combinations that actually exist (for the app's dropdowns)."""
    recent = df[df["year"].between(year - 2, year) & df["is_complete_year"]]
    cols = ["job_code", "job", "job_family", "department_code", "department",
            "organization_group", "union_code", "union"]
    return (recent.groupby(cols).size().rename("records").reset_index()
            .sort_values(["job", "records"], ascending=[True, False]))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    year = int(df.loc[df["is_complete_year"], "year"].max())

    trend = pay_trend(df)
    ot = overtime_by_department(df, year)
    gaps, gap_detail = same_job_gaps(df, year)
    lookup = job_lookup(df, year)
    combo = combos(df, year)

    trend.to_csv(OUT / "pay_trend.csv", index=False)
    ot.to_csv(OUT / "overtime_by_department.csv", index=False)
    gaps.to_csv(OUT / "same_job_gaps.csv", index=False)
    gap_detail.to_csv(OUT / "same_job_gap_detail.csv", index=False)
    lookup.to_csv(OUT / "job_lookup.csv", index=False)
    combo.to_csv(OUT / "combos.csv", index=False)

    first, last = trend.iloc[0], trend.iloc[-1]
    yrs = int(last["year"] - first["year"])
    cagr = lambda a, b: (b / a) ** (1 / yrs) - 1
    headline = {
        "latest_complete_year": year,
        "first_year": int(first["year"]),
        "incomplete_years": sorted(df.loc[~df["is_complete_year"], "year"].unique().tolist()),
        "full_year_employees_latest": int(last["employees"]),
        "median_total_latest": float(last["median_total"]),
        "median_total_first": float(first["median_total"]),
        "growth_total_pct": float(last["median_total"] / first["median_total"] - 1),
        "cagr_total": float(cagr(first["median_total"], last["median_total"])),
        "cagr_base": float(cagr(first["median_base"], last["median_base"])),
        "cagr_benefits": float(cagr(first["median_benefits"], last["median_benefits"])),
        "overtime_share_latest": float(last["overtime_share_of_cash_pay"]),
        "overtime_share_first": float(first["overtime_share_of_cash_pay"]),
        "share_with_overtime_latest": float(last["share_with_overtime"]),
        "top_ot_departments": ot.head(3)[["department", "overtime_share_of_cash_pay"]].to_dict("records"),
        "top_gap_job": gaps.head(1).to_dict("records"),
        "jobs_in_multiple_departments": int(len(gaps)),
        "median_gap_pct_total": float(gaps["total_gap_pct"].median()),
        "median_gap_pct_base": float(gaps["base_gap_pct"].median()),
    }
    (OUT / "headline.json").write_text(json.dumps(headline, indent=2, default=float))
    print(json.dumps(headline, indent=2, default=float))
    print(trend.round(3).to_string())
    print(ot.head(8).round(3).to_string())
    print(gaps.head(10).round(2).to_string())


if __name__ == "__main__":
    main()
