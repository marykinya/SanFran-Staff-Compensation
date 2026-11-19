"""San Francisco public pay: a scrolling story.

Walks through how City and County of San Francisco pay has changed, then lets the
reader estimate what a role pays. Data: DataSF Employee Compensation.
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

BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY, LIGHT_GREY, INK = "#8a8985", "#d9d7d0", "#1f1e1c"
FEATURE_LABELS = {
    "job_code": "Job title", "job_family": "Job family", "department_code": "Department",
    "organization_group": "Organisation group", "union_code": "Union", "year": "Year",
}

st.set_page_config(page_title="SF Public Pay", page_icon="💼", layout="centered")

st.markdown(
    """
    <style>
      .block-container {max-width: 760px; padding-top: 3rem;}
      .kicker {color: #2a78d6; font-size: 0.8rem; font-weight: 700;
               letter-spacing: 0.12em; text-transform: uppercase; margin: 0;}
      .big {font-size: clamp(2.6rem, 9vw, 4.2rem); font-weight: 800; line-height: 1.05;
            color: #1f1e1c; margin: 0.2rem 0 0.4rem;}
      .lede {font-size: 1.2rem; line-height: 1.6; color: #3d3c39;}
      .chapter {margin-top: 5.5rem;}
      .note {color: #6b6a66; font-size: 0.9rem;}
      .scroll-hint {color: #8a8985; text-align: center; margin-top: 3rem; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


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
FIRST, LATEST = H["first_year"], H["latest_complete_year"]


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
            "effects": effects}


def money(x: float) -> str:
    return f"${x:,.0f}"


def k(x: float) -> str:
    return f"${x / 1000:,.0f}k"


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      template="plotly_white", font=dict(color=INK, size=13),
                      plot_bgcolor="white", paper_bgcolor="white",
                      hoverlabel=dict(namelength=-1, bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor=LIGHT_GREY)
    fig.update_yaxes(gridcolor="#eeede8", zeroline=False)
    return fig


def chart(fig: go.Figure, height: int = 360):
    st.plotly_chart(style(fig, height), width="stretch", config={"displayModeBar": False})


def chapter(kicker: str, big: str, lede: str):
    st.markdown(
        f'<div class="chapter"><p class="kicker">{kicker}</p>'
        f'<p class="big">{big}</p><p class="lede">{lede}</p></div>',
        unsafe_allow_html=True,
    )


trend, ot, gaps, detail = T["trend"], T["overtime"], T["gaps"], T["gap_detail"]

# ------------------------------------------------------------------ opening
st.markdown('<p class="kicker">San Francisco public pay</p>', unsafe_allow_html=True)
st.markdown("# What does a city job in San Francisco really pay?")
st.markdown(
    f'<p class="lede">The typical full-year city employee cost '
    f'<b>{money(H["median_total_latest"])}</b> in FY{LATEST}, counting salary, overtime and benefits. '
    f'That is <b>{H["growth_total_pct"]:.0%} more</b> than in FY{FIRST}. '
    f"Here's where that money goes, and what it means for a single role.</p>",
    unsafe_allow_html=True,
)
st.markdown('<p class="scroll-hint">Scroll to read ↓</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ 1. benefits
chapter(
    "Chapter 1 · Benefits",
    f"{H['cagr_benefits']:.1%} a year",
    f"That's how fast benefits grew for the typical employee. Base salary grew "
    f"{H['cagr_base']:.1%} a year. The gap looks small, but it compounds: benefits now "
    f"take a bigger share of what each job costs the city.",
)
fig = go.Figure()
for col, name, color in [("index_base", "Base salary", GREY), ("index_benefits", "Benefits", ORANGE)]:
    fig.add_trace(go.Scatter(x=trend["year"], y=trend[col], name=name, mode="lines",
                             line=dict(color=color, width=3),
                             hovertemplate=f"{name}<br>FY%{{x}}: %{{y:.0f}}<extra></extra>"))
    last = trend.iloc[-1]
    fig.add_annotation(x=last["year"], y=last[col], text=f"<b>{name}</b><br>+{last[col] - 100:.0f}%",
                       showarrow=False, xanchor="left", xshift=8, font=dict(color=color))
fig.update_xaxes(dtick=2, range=[FIRST - 0.3, LATEST + 1.6])
fig.update_yaxes(title=f"Growth since FY{FIRST} (FY{FIRST} = 100)")
chart(fig)

# ------------------------------------------------------------------ 2. overtime
top3 = ot.head(3)
chapter(
    "Chapter 2 · Overtime",
    "1 in 5 dollars",
    f"Most city workers ({H['share_with_overtime_latest']:.0%}) earn some overtime, but it adds up to "
    f"only {H['overtime_share_latest']:.1%} of cash pay city-wide. Public safety is different. At the "
    + ", ".join(f"<b>{d}</b>" for d in top3["department"][:2])
    + f" and <b>{top3['department'].iloc[2]}</b>, one dollar in every five or six is overtime.",
)
show = ot.head(10).iloc[::-1]
colors = [ORANGE if d in set(top3["department"]) else LIGHT_GREY for d in show["department"]]
fig = go.Figure(go.Bar(
    x=show["overtime_share_of_cash_pay"], y=show["department"], orientation="h", marker_color=colors,
    text=[f"{v:.0%}" for v in show["overtime_share_of_cash_pay"]], textposition="outside",
    customdata=show[["overtime_total_musd"]],
    hovertemplate="%{y}<br>Overtime: %{x:.1%} of cash pay ($%{customdata[0]:.0f}M)<extra></extra>"))
fig.update_xaxes(visible=False, range=[0, ot["overtime_share_of_cash_pay"].max() * 1.18])
fig.update_yaxes(showgrid=False)
chart(fig, 400)
st.markdown(f'<p class="note">Overtime as a share of cash pay, top 10 departments, FY{LATEST}.</p>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 3. same job
tg = H["top_gap_job"][0]
chapter(
    "Chapter 3 · Same job, same pay",
    f"{H['median_gap_pct_total']:.1%}",
    f"That's the typical pay gap for the same job title across different departments. "
    f"San Francisco pays by job classification, not by department. But there are exceptions. "
    f"A <b>{tg['job']}</b> earns {k(tg['low_total'])} to {k(tg['high_total'])} depending on "
    f"where they work.",
)
options = gaps.head(12)
labels = (options["job"] + "  ·  " + options["job_code"]).tolist()
pick = st.selectbox("Explore another job title", labels, key="gap_job")
code = pick.rsplit("·", 1)[1].strip()
d = detail[detail["job_code"] == code].sort_values("median_total")
fig = go.Figure(go.Bar(
    y=d["department"], x=d["median_total"], orientation="h",
    marker_color=[ORANGE if i == len(d) - 1 else LIGHT_GREY for i in range(len(d))],
    text=[k(v) for v in d["median_total"]], textposition="outside", customdata=d["n"],
    hovertemplate="%{y}<br>Median total comp: %{x:$,.0f}<br>%{customdata} employees<extra></extra>"))
fig.update_xaxes(visible=False, range=[0, d["median_total"].max() * 1.15])
fig.update_yaxes(showgrid=False)
chart(fig, 90 + 55 * len(d))
st.markdown('<p class="note">Median total compensation by department.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ 4. estimator
chapter(
    "Your turn",
    "What would a role pay?",
    "Pick a job, department and union. A model trained on city pay records estimates total "
    "compensation for that role. It works from the role alone, never from any one person's pay.",
)
combos = T["combos"]
jobs = combos.groupby(["job_code", "job"])["records"].sum().reset_index()
jobs["label"] = jobs["job"] + "  ·  " + jobs["job_code"]
default_job = jobs.index[jobs["job_code"] == "1822"]
job_label = st.selectbox("Job title", jobs["label"],
                         index=int(default_job[0]) if len(default_job) else 0,
                         help="Start typing to search 1,000+ job titles")
job_code = job_label.rsplit("·", 1)[1].strip()

jc = combos[combos["job_code"] == job_code]
c1, c2 = st.columns(2)
depts = jc.groupby(["department_code", "department"])["records"].sum().sort_values(ascending=False)
dept_name = c1.selectbox("Department", [dn for _, dn in depts.index])
jd = jc[jc["department"] == dept_name]
unions = jd.groupby(["union_code", "union"])["records"].sum().sort_values(ascending=False)
union_name = c2.selectbox("Union", [u for _, u in unions.index])
jdu = jd[jd["union"] == union_name].iloc[0]

year = st.select_slider("Fiscal year", list(range(LATEST - 2, LATEST + 4)), value=LATEST + 1)

est = estimate({"job_code": job_code, "job_family": jdu["job_family"],
                "department_code": jdu["department_code"],
                "organization_group": jdu["organization_group"],
                "union_code": jdu["union_code"], "year": year})

projection = f" (a projection, extending the FY{FIRST}-{LATEST} trend)" if year > LATEST else ""
st.markdown(
    f'<p class="big" style="margin-top:2rem">{money(est["p50"])}</p>'
    f'<p class="lede">Estimated total compensation in FY{year}{projection}. '
    f'Most people in this role (8 in 10) would fall between <b>{k(est["p10"])}</b> and '
    f'<b>{k(est["p90"])}</b>.</p>',
    unsafe_allow_html=True,
)

look = T["lookup"].set_index("job_code")
if job_code in look.index:
    lk = look.loc[job_code]
    fig = go.Figure()
    rows = [("This estimate", est["p10"], est["p90"], est["p50"], BLUE),
            (f"Actual, FY{lk['source_years']}", lk["p10"], lk["p90"], lk["p50"], GREY)]
    for name, lo, hi, mid, color in rows:
        fig.add_trace(go.Scatter(x=[lo, hi], y=[name, name], mode="lines",
                                 line=dict(color=color, width=14), opacity=0.3, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[mid], y=[name], mode="markers+text", text=[k(mid)],
                                 textposition="top center", marker=dict(color=color, size=16),
                                 hovertemplate=f"{name}<br>Median %{{x:$,.0f}}<br>"
                                               f"Range {money(lo)} to {money(hi)}<extra></extra>"))
    fig.update_xaxes(tickformat="$,.0s")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    chart(fig, 190)
    st.markdown(
        f'<p class="note">Shaded bars show the range for 8 in 10 people. "Actual" covers '
        f"{int(lk['n']):,} full-year {lk['job']} employees across all departments.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("How this pay breaks down, and what moves the estimate"):
        st.markdown(f"**A typical {lk['job']}** (medians, FY{lk['source_years']})")
        b1, b2, b3 = st.columns(3)
        b1.metric("Base salary", money(lk["median_base"]))
        b2.metric("Overtime", money(lk["median_overtime"]))
        b3.metric("Benefits", money(lk["median_benefits"]))

        st.markdown("**What moves this estimate**, compared with the average city role")
        eff = (np.exp(est["effects"].rename(index=FEATURE_LABELS)) - 1).sort_values()
        eff = eff[eff.abs() >= 0.005]
        fig = go.Figure(go.Bar(
            x=eff.values, y=eff.index, orientation="h",
            marker_color=[ORANGE if v < 0 else BLUE for v in eff.values],
            text=[f"{v:+.0%}" for v in eff.values], textposition="outside",
            hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
        fig.update_xaxes(tickformat="+.0%")
        fig.update_yaxes(showgrid=False)
        chart(fig, 60 + 45 * len(eff))
        st.caption("Blue pushes the estimate up, orange pulls it down. Job title and union often "
                   "overlap, because many unions cover one occupation group.")

# ------------------------------------------------------------------ 5. fine print
ev = meta["eval"]
comp = T["comparison"]
ours = comp.loc["LightGBM (this model)"]
base = comp.loc["Job title median (last year, grown)"]

chapter(
    "The fine print",
    f"{ours['R2']:.0%}",
    f"That's how much of the variation in pay the role explains. The rest is personal: "
    f"overtime hours, seniority steps, time on leave. No role-level model can know those.",
)
with st.expander("How the model was built and tested"):
    st.markdown(
        f"* **Inputs:** job title, job family, department, organisation group, union and year. "
        "Pay components are never used as inputs, because they add up exactly to the answer.\n"
        f"* **Tested on the future:** trained on FY{ev['train_years'][0]}-{ev['train_years'][1]}, "
        f"tested on FY{ev['test_year']} ({ev['n_test']:,} full-year employees it never saw).\n"
        "* **Two parts:** a city-wide pay index tracks pay rises over time, and a LightGBM model "
        "learns each role's pay relative to it.\n"
        f"* **Honest ranges:** {ours['coverage_80']:.0%} of FY{ev['test_year']} employees fell inside "
        "the 80% range.\n"
        f"* **A simple benchmark does about as well:** last year's median for each job title scores "
        f"R² {base['R2']:.3f}, versus {ours['R2']:.3f} for the model. That itself confirms SF pays by "
        "job classification. The model adds calibrated ranges and estimates for any job, "
        "department and union combination."
    )
    show = comp.copy()
    show.columns = ["R²", "Mean abs. error", "Median abs. % error", "Within 10%", "80% range covers"]
    fmt = {"R²": "{:.3f}", "Mean abs. error": "${:,.0f}", "Median abs. % error": "{:.1%}",
           "Within 10%": "{:.0%}", "80% range covers": "{:.0%}"}
    for col, f in fmt.items():
        show[col] = show[col].map(lambda v, f=f: "n/a" if pd.isna(v) else f.format(v))
    st.dataframe(show, width="stretch")

with st.expander("Limitations"):
    inc = ", ".join(f"FY{y}" for y in H["incomplete_years"]) or "none"
    st.markdown(
        f"* **Incomplete years excluded:** {inc}. In the source export, health & dental was $0 for "
        "about 93% of FY2022 records, which would make FY2022 look like a pay cut.\n"
        "* **No hours worked.** \"Full year\" is a proxy: base salary at least half the job's median "
        "that year, and at least $15k.\n"
        "* **No seniority or individual history.** That is deliberate, since the estimate is for a "
        "role, not a person.\n"
        "* **Projections assume the historical pay trend continues.**"
    )

st.markdown('<div class="chapter"></div>', unsafe_allow_html=True)
st.divider()
st.caption(
    "Data: [DataSF Employee Compensation](https://data.sfgov.org/d/88g8-5mnd) · "
    "Built by [Mary Kinya](https://marykinya.com) · "
    "[Code on GitHub](https://github.com/marykinya/SanFran-Staff-Compensation)"
)
