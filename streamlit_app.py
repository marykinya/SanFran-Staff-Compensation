"""San Francisco public pay explorer.

Estimates what a City and County of San Francisco role pays, from the role alone,
and explores how city pay has changed. Data: DataSF Employee Compensation.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
APP_DATA = ROOT / "app_data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#8a8985"
FEATURE_LABELS = {
    "job_code": "Job title", "job_family": "Job family", "department_code": "Department",
    "organization_group": "Organisation group", "union_code": "Union", "year": "Year",
}

st.set_page_config(page_title="SF Public Pay Explorer", page_icon="💼", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_models():
    models = {k: lgb.Booster(model_file=str(MODELS / f"lgbm_{k}.txt")) for k in ("p10", "p50", "p90")}
    meta = json.loads((MODELS / "metadata.json").read_text())
    return models, meta


@st.cache_data
def load_tables():
    read = lambda name, **kw: pd.read_csv(APP_DATA / name, **kw)
    codes = {"job_code": str, "department_code": str, "union_code": str}
    return {
        "combos": read("combos.csv", dtype=codes),
        "lookup": read("job_lookup.csv", dtype={"job_code": str}),
        "trend": read("pay_trend.csv"),
        "overtime": read("overtime_by_department.csv"),
        "gaps": read("same_job_gaps.csv", dtype={"job_code": str}),
        "gap_detail": read("same_job_gap_detail.csv", dtype={"job_code": str}),
        "comparison": pd.read_csv(REPORTS / "model_comparison.csv", index_col=0),
        "importance": pd.read_csv(REPORTS / "feature_importance.csv"),
        "headline": json.loads((APP_DATA / "headline.json").read_text()),
    }


models, meta = load_models()
T = load_tables()
H = T["headline"]
LATEST = H["latest_complete_year"]


def pay_index(year: int) -> float:
    idx = meta["pay_index"]
    if str(year) in idx["medians"]:
        return idx["medians"][str(year)]
    return float(np.exp(idx["intercept"] + idx["slope"] * year))


def estimate(row: dict) -> dict:
    X = pd.DataFrame([row])
    for c in meta["categorical_features"]:
        X[c] = pd.Categorical(X[c].astype(str))
    X = X[meta["features"]]
    raw = {k: float(m.predict(X)[0]) for k, m in models.items()}
    w = meta["band_widen"]
    mid = raw["p50"]
    lo = mid - w * max(mid - raw["p10"], 0)
    hi = mid + w * max(raw["p90"] - mid, 0)
    level = pay_index(row["year"])
    contrib = models["p50"].predict(X, pred_contrib=True)[0]
    effects = pd.Series(contrib[:-1], index=meta["features"])
    return {"p10": level * np.exp(lo), "p50": level * np.exp(mid), "p90": level * np.exp(hi),
            "level": level, "base": contrib[-1], "effects": effects}


def money(x: float) -> str:
    return f"${x:,.0f}"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08, x=0), hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ------------------------------------------------------------------ header
st.title("San Francisco Public Pay Explorer")
st.caption(
    f"What does a City of San Francisco job really pay? Built on {H['first_year']}-{LATEST} "
    "fiscal-year records from DataSF's Employee Compensation dataset."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Median total comp, FY{LATEST}", money(H["median_total_latest"]),
          f"+{H['growth_total_pct']:.0%} since FY{H['first_year']}")
c2.metric("Benefits growth per year", f"{H['cagr_benefits']:.1%}",
          f"vs {H['cagr_base']:.1%} for base pay", delta_color="off", delta_arrow="off")
c3.metric("Employees earning overtime", f"{H['share_with_overtime_latest']:.0%}",
          f"overtime = {H['overtime_share_latest']:.1%} of cash pay", delta_color="off", delta_arrow="off")
c4.metric("Full-year employees", f"{H['full_year_employees_latest']:,}", f"FY{LATEST}", delta_color="off", delta_arrow="off")

tab_est, tab_ins, tab_model = st.tabs(["🔮 Pay estimator", "📊 Insights", "🧪 How the model works"])

# ------------------------------------------------------------------ estimator
with tab_est:
    st.markdown(
        "Pick a role. The model estimates total compensation (salary, overtime and benefits) "
        "**from the role alone**. It never sees what any individual was actually paid."
    )
    combos = T["combos"]
    left, right = st.columns([1, 2], gap="large")

    with left:
        jobs = combos.groupby(["job_code", "job"])["records"].sum().reset_index()
        jobs["label"] = jobs["job"] + "  ·  " + jobs["job_code"]
        default_job = jobs.index[jobs["job_code"] == "1822"]
        job_label = st.selectbox("Job title", jobs["label"],
                                 index=int(default_job[0]) if len(default_job) else 0,
                                 help="Start typing to search 1,000+ job titles")
        job_code = job_label.rsplit("·", 1)[1].strip()

        jc = combos[combos["job_code"] == job_code]
        depts = jc.groupby(["department_code", "department"])["records"].sum().sort_values(ascending=False)
        dept_name = st.selectbox("Department", [d for _, d in depts.index])
        jd = jc[jc["department"] == dept_name]
        unions = jd.groupby(["union_code", "union"])["records"].sum().sort_values(ascending=False)
        union_name = st.selectbox("Union", [u for _, u in unions.index])
        jdu = jd[jd["union"] == union_name].iloc[0]

        years = list(range(LATEST - 2, LATEST + 4))
        year = st.select_slider("Fiscal year", years, value=LATEST + 1)
        if year > LATEST:
            st.caption(f"FY{year} is a projection: city-wide pay is extended on its "
                       f"{H['first_year']}-{LATEST} trend.")

    row = {"job_code": job_code, "job_family": jdu["job_family"],
           "department_code": jdu["department_code"],
           "organization_group": jdu["organization_group"],
           "union_code": jdu["union_code"], "year": year}
    est = estimate(row)

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated total comp", money(est["p50"]))
        # "\$" stops Streamlit reading a pair of dollar signs as LaTeX
        m2.metric("Likely range (80%)", f"\\${est['p10'] / 1000:,.0f}k to \\${est['p90'] / 1000:,.0f}k")
        look = T["lookup"].set_index("job_code")
        if job_code in look.index:
            lk = look.loc[job_code]
            m3.metric(f"Actual median, FY{lk['source_years']}", money(lk["p50"]),
                      f"{int(lk['n']):,} full-year employees", delta_color="off", delta_arrow="off")

            # range chart: actual spread vs estimate band
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=["Actual pay<br>(all departments)"], x=[lk["p90"] - lk["p10"]], base=[lk["p10"]],
                orientation="h", marker=dict(color=GREY, opacity=0.35), width=0.45,
                name="Actual: 10th-90th percentile",
                hovertemplate=f"10th pct {money(lk['p10'])}<br>90th pct {money(lk['p90'])}<extra></extra>"))
            fig.add_trace(go.Bar(
                y=["Actual pay<br>(all departments)"], x=[lk["p75"] - lk["p25"]], base=[lk["p25"]],
                orientation="h", marker=dict(color=GREY, opacity=0.8), width=0.45,
                name="Actual: middle 50%",
                hovertemplate=f"25th pct {money(lk['p25'])}<br>75th pct {money(lk['p75'])}<extra></extra>"))
            fig.add_trace(go.Bar(
                y=["Estimate<br>(this role)"], x=[est["p90"] - est["p10"]], base=[est["p10"]],
                orientation="h", marker=dict(color=BLUE, opacity=0.35), width=0.45,
                name="Estimate: 80% range",
                hovertemplate=f"{money(est['p10'])} to {money(est['p90'])}<extra></extra>"))
            fig.add_trace(go.Scatter(
                y=["Estimate<br>(this role)", "Actual pay<br>(all departments)"],
                x=[est["p50"], lk["p50"]], mode="markers",
                marker=dict(color=[BLUE, "#52514e"], size=14, symbol="line-ns",
                            line=dict(width=3, color=[BLUE, "#52514e"])),
                name="Median", hovertemplate="%{x:$,.0f}<extra></extra>"))
            fig.update_layout(barmode="overlay", showlegend=True)
            fig.update_xaxes(tickformat="$,.0f", title=None)
            st.plotly_chart(style(fig, 260), width="stretch")

            comp = pd.DataFrame({
                "Component": ["Base salary", "Overtime", "Benefits"],
                "Typical amount": [money(v) for v in
                                   (lk["median_base"], lk["median_overtime"], lk["median_benefits"])],
            })
            st.markdown(f"**What a typical *{lk['job']}* is paid, by component** (medians, FY{lk['source_years']})")
            st.dataframe(comp, hide_index=True, width="stretch")

        st.markdown("**What moves this estimate** (compared with the average city role)")
        eff = est["effects"].rename(index=FEATURE_LABELS)
        eff = (np.exp(eff) - 1).sort_values()
        eff = eff[eff.abs() >= 0.005]
        fig = go.Figure(go.Bar(
            x=eff.values, y=eff.index, orientation="h",
            marker_color=[ORANGE if v < 0 else BLUE for v in eff.values],
            text=[f"{v:+.0%}" for v in eff.values], textposition="outside",
            hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
        fig.update_xaxes(tickformat="+.0%", title=None)
        st.plotly_chart(style(fig, 60 + 45 * len(eff)), width="stretch")
        st.caption("Each bar is that input's contribution to the estimate (SHAP-style "
                   "contributions from the LightGBM model). Job title and union tend to overlap, "
                   "because many unions cover one occupation group.")

# ------------------------------------------------------------------ insights
with tab_ins:
    trend, ot, gaps, detail = T["trend"], T["overtime"], T["gaps"], T["gap_detail"]

    st.subheader("1. Benefits are growing faster than pay")
    st.markdown(
        f"Between FY{H['first_year']} and FY{LATEST}, the median full-year employee's base salary "
        f"grew **{H['cagr_base']:.1%} a year** and benefits grew **{H['cagr_benefits']:.1%} a year**. "
        "Benefits now make up a growing share of what each role costs the city."
    )
    fig = go.Figure()
    for col, name, color in [("index_base", "Base salary", BLUE),
                             ("index_benefits", "Benefits", ORANGE),
                             ("index_total", "Total compensation", AQUA)]:
        fig.add_trace(go.Scatter(x=trend["year"], y=trend[col], name=name, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=8),
                                 hovertemplate=f"{name}<br>FY%{{x}}: %{{y:.0f}}<extra></extra>"))
    fig.update_yaxes(title=f"Index (FY{H['first_year']} = 100)")
    fig.update_xaxes(dtick=1, title=None)
    st.plotly_chart(style(fig), width="stretch")

    st.subheader("2. Overtime is concentrated in public safety")
    top = ot.head(3)
    st.markdown(
        f"Overtime is {H['overtime_share_latest']:.1%} of the city's cash pay overall. In "
        + ", ".join(f"**{r.department}** ({r.overtime_share_of_cash_pay:.0%})" for r in top.itertuples())
        + " it is roughly one dollar in every five or six. "
        f"City-wide it rose from {H['overtime_share_first']:.1%} in FY{H['first_year']} "
        f"to a peak of {trend['overtime_share_of_cash_pay'].max():.1%} in "
        f"FY{int(trend.loc[trend['overtime_share_of_cash_pay'].idxmax(), 'year'])}."
    )
    show = ot.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=show["overtime_share_of_cash_pay"], y=show["department"], orientation="h",
        marker_color=BLUE, customdata=show[["employees", "overtime_total_musd", "share_ot_over_25pct_of_base"]],
        hovertemplate=("%{y}<br>Overtime share of cash pay: %{x:.1%}<br>"
                       "Overtime paid: $%{customdata[1]:.1f}M<br>"
                       "Employees with OT > 25% of base: %{customdata[2]:.0%}<br>"
                       "Full-year employees: %{customdata[0]:,}<extra></extra>")))
    fig.update_xaxes(tickformat=".0%", title=f"Overtime as a share of cash pay, FY{LATEST}")
    st.plotly_chart(style(fig, 480), width="stretch")

    st.subheader("3. Same job, same pay. Almost.")
    st.markdown(
        f"{H['jobs_in_multiple_departments']} job titles have at least 15 full-year employees in two or "
        f"more departments. For the typical one, the median pay gap between departments is only "
        f"**{H['median_gap_pct_total']:.1%}**. SF pays by job classification, not by department. "
        "The exceptions are worth a look: some come from overtime, others from different seniority mixes."
    )
    options = gaps.head(12)
    pick = st.selectbox("Job title", options["job"] + "  ·  " + options["job_code"], key="gap_job")
    code = pick.rsplit("·", 1)[1].strip()
    d = detail[detail["job_code"] == code].sort_values("median_total")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["department"], x=d["median_base"], name="Median base salary",
                         orientation="h", marker_color=BLUE,
                         hovertemplate="%{y}<br>Base: %{x:$,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(y=d["department"], x=d["median_total"], name="Median total comp",
                         orientation="h", marker_color=ORANGE,
                         customdata=d["n"],
                         hovertemplate="%{y}<br>Total: %{x:$,.0f}<br>%{customdata} employees<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_xaxes(tickformat="$,.0f", title=None)
    st.plotly_chart(style(fig, 140 + 70 * len(d)), width="stretch")

# ------------------------------------------------------------------ model
with tab_model:
    ev = meta["eval"]
    comp = T["comparison"]
    ours = comp.loc["LightGBM (this model)"]
    base = comp.loc["Job title median (last year, grown)"]

    st.subheader("The question")
    st.markdown(
        "Given only what you know about a role (job title, job family, department, organisation "
        "group, union and year), what total compensation should you expect?\n\n"
        "The six pay components (salary, overtime, other salary, retirement, health & dental, other "
        "benefits) add up *exactly* to total compensation, so none of them are used as inputs. "
        "A model given those columns scores R² ≈ 0.999, but it has only learned addition."
    )

    st.subheader("How it was tested")
    st.markdown(
        f"* **Trained on FY{ev['train_years'][0]}-{ev['train_years'][1]}, tested on FY{ev['test_year']}.** "
        f"A random split would let the same person appear in training and test across years.\n"
        f"* **Full-year employees only** ({ev['n_train']:,} training rows, {ev['n_test']:,} test rows). "
        "Part-year and part-time records are kept in the data but flagged.\n"
        "* **Pay level and pay structure are modelled separately.** A city-wide pay index captures "
        "inflation and pay rises (projected forward on its trend). A LightGBM model learns each "
        "role's pay relative to that index.\n"
        "* **Ranges come from quantile models** (10th and 90th percentile), calibrated on the year "
        "before the test year so the band really covers about 80% of people."
    )

    st.subheader(f"Results on FY{ev['test_year']} (never seen in training)")
    show = comp.copy()
    show.columns = ["R²", "Mean abs. error", "Median abs. % error", "Within 10%", "80% range covers"]
    fmt = {"R²": "{:.3f}", "Mean abs. error": "${:,.0f}", "Median abs. % error": "{:.1%}",
           "Within 10%": "{:.0%}", "80% range covers": "{:.0%}"}
    for col, f in fmt.items():
        show[col] = show[col].map(lambda v, f=f: "n/a" if pd.isna(v) else f.format(v))
    st.dataframe(show, width="stretch")
    st.markdown(
        f"**The honest takeaway:** the role explains about **{ours['R2']:.0%}** of the variation in pay. "
        f"A simple lookup of last year's median for each job title does almost exactly as well "
        f"(R² {base['R2']:.3f} vs {ours['R2']:.3f}). That is a finding in itself: SF pay is set by "
        "job classification, and most of what's left is individual (overtime hours, seniority steps, "
        "time on leave), which no role-level model can know.\n\n"
        f"What the model adds is a **calibrated range** ({ours['coverage_80']:.0%} of FY{ev['test_year']} "
        "employees fell inside it), estimates for job, department and union combinations, and a "
        "breakdown of what drives each estimate."
    )

    st.subheader("What drives pay")
    imp = T["importance"].iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["share"], y=imp["feature"], orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.0%} of total effect<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="Share of the model's total effect (mean |contribution|)")
    st.plotly_chart(style(fig, 300), width="stretch")

    st.subheader("Limitations")
    inc = ", ".join(f"FY{y}" for y in H["incomplete_years"]) or "none"
    st.markdown(
        f"* **Incomplete years excluded:** {inc}. In the source export, health & dental was $0 for "
        "about 93% of FY2022 records, which would make FY2022 look like a pay cut.\n"
        "* **No hours worked.** \"Full year\" is a proxy: base salary at least half the job's median "
        "that year, and at least $15k.\n"
        "* **No seniority, tenure or individual history.** That is deliberate (the estimate is for a "
        "role, not a person) and it is the main reason R² stops near 0.78.\n"
        "* **Projections assume the historical pay trend continues.**"
    )

st.divider()
st.caption(
    "Data: [DataSF Employee Compensation](https://data.sfgov.org/d/88g8-5mnd) · "
    "Built by [Mary Kinya](https://marykinya.com) · "
    "[Code on GitHub](https://github.com/marykinya/SanFran-Staff-Compensation)"
)
