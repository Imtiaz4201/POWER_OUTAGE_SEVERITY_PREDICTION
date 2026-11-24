"""Build final_stage2/exp1_walk_forward.ipynb: v2 data pipeline (unchanged) + Experiment 1 (walk-forward by year)."""
import json, sys
SRC = "/Users/imtiaz/Documents/po1/progress_stage2/orginalProblemStatement_outage_extended_noaa_newgridv2.ipynb"
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp1_walk_forward/exp1_walk_forward.ipynb"
nb = json.load(open(SRC))
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}
src = lambda c: "".join(c["source"])

INTRO = """# Final stage, Experiment 1: walk-forward evaluation by year

**Problem statement:** *Can we predict how severe a major power outage will become, using historical outage data and recent storm information available when the outage starts?*

**Question of this experiment:** how stable is the model over time, and is 2024 different from earlier years?

**Design (decided before running):**
- Data: extended outage data 2000–2024 + NOAA storms + NWS warnings (base model, no infrastructure data). The data pipeline (sections 0–6) is unchanged from the progress-stage notebook; all four storm rules are built. Outputs go to `results/` next to this notebook.
- Setup: **strict storm rule (R2)**, **scenario A (pre-event)**.
- **Expanding window:** for each 2-year test block (2008–09, 2010–11, …, 2022–23) the model is trained on **all years before the block** and tested on the block.
- **2024 is evaluated separately** (train 2000–2023, test 2024), because the OE-417 form changed in 2024. Section 9 compares 2024 with earlier years on the data itself.
- One fixed model per target (random forest, moderate settings, recency weights), so differences between blocks come from time, not from tuning. Model comparison and tuning are Experiment 2.
- Severe labels: Q1 / Q3 from each block's training years only.
"""
cells = [md(INTRO)]
for i, c in enumerate(nb["cells"][1:42], start=1):
    c = json.loads(json.dumps(c)); c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs")
    if c["cell_type"] == "code":
        c["execution_count"] = None
        s = src(c)
        if s.startswith("ROOT = ("):
            s = s.replace('ROOT = (\n    os.getcwd()\n    if os.path.isdir(os.path.join(os.getcwd(), "Data"))\n    else os.path.dirname(os.getcwd())\n)',
                          '# project folder = the first parent folder that contains Data/\nROOT = os.getcwd()\nwhile not os.path.isdir(os.path.join(ROOT, "Data")) and os.path.dirname(ROOT) != ROOT:\n    ROOT = os.path.dirname(ROOT)')
            s = s.replace('OUT_DIR = os.path.join(\n    ROOT, "progress_stage2", "results_outage_extended_noaa_nws_newgrid"\n)  # CSV / PNG outputs',
                          'OUT_DIR = os.path.join(os.getcwd(), "results")  # CSV / PNG outputs, next to this notebook')
            assert 'while not os.path.isdir' in s and 'os.getcwd(), "results"' in s
            s = s.replace('TEST_START = pd.Timestamp("2022-01-01")  # test period = 2022-2024',
                          'TEST_START = pd.Timestamp("2022-01-01")  # used only for the summary printout in section 6')
            c["source"] = s.splitlines(keepends=True)
    if c["cell_type"] == "markdown":
        c.pop("outputs", None)
    cells.append(c)

cells.append(md("""## 7. Experiment 1: walk-forward evaluation

Expanding window, 2-year test blocks, strict rule, scenario A. Metrics on each test block:
- customers and duration: R², MAE (log scale) and Spearman rank correlation;
- is_severe: ROC-AUC, PR-AUC and the share of severe outages in the block (PR-AUC must be read against it)."""))
cells.append(code('''from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score

WF_RULE, WF_SCEN = "strict", "A_pre_event"
WF_BLOCKS = [(y, y + 1) for y in range(2008, 2024, 2)]  # 2008-09, 2010-11, ..., 2022-23
WF_2024 = (2024, 2024)  # evaluated separately
WF_PRED = {}  # (test block, task) -> (true values, predictions), used for the bootstrap in section 10


def wf_model(kind):
    """One fixed model per target (no tuning here)."""
    if kind == "reg":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.33,
                                     random_state=SEED, n_jobs=-1)
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.33,
                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def wf_block(df, y0, y1, num, cat):
    """Train on all years before y0, test on y0..y1."""
    yr = df["outage_start"].dt.year
    train, test = df[yr < y0], df[(yr >= y0) & (yr <= y1)]
    row = {"test_years": f"{y0}-{y1}" if y1 > y0 else str(y0), "train_years": f"2000-{y0 - 1}",
           "n_train": len(train), "n_test": len(test)}
    for task, (kind, target) in TASKS.items():
        d_tr, d_te = task_rows(train, kind, target), task_rows(test, kind, target)
        y_tr, y_te = make_labels(kind, target, train, d_tr, d_te)
        if len(d_te) < 10 or (kind == "bin" and len(np.unique(y_te)) < 2):
            continue
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("model", wf_model(kind))])
        pipe.fit(d_tr[num + cat], y_tr, model__sample_weight=recency_weights(d_tr))
        p = predict(pipe, kind, d_te[num + cat])
        WF_PRED[(row["test_years"], task)] = (np.asarray(y_te), np.asarray(p))
        if kind == "reg":
            k = task.replace("_reg", "")
            row.update({f"{k}_n": len(y_te), f"{k}_R2": r2_score(y_te, p), f"{k}_MAE": mean_absolute_error(y_te, p),
                        f"{k}_rho": spearmanr(y_te, p).correlation})
        else:
            row.update({"severe_n": len(y_te), "severe_share": y_te.mean(),
                        "severe_ROC_AUC": roc_auc_score(y_te, p), "severe_PR_AUC": average_precision_score(y_te, p)})
    return row


t0 = time.time()
df = tables[WF_RULE]
num, cat = feature_sets(df, WF_SCEN)
wf = pd.DataFrame([wf_block(df, y0, y1, num, cat) for y0, y1 in WF_BLOCKS + [WF_2024]])
wf.to_csv(os.path.join(OUT_DIR, "exp1_walk_forward.csv"), index=False)
print(f"walk-forward done in {time.time() - t0:.0f}s")
display(wf.round(3))'''))

cells.append(md("""### 7.1 Summary: 2008–2023 blocks vs 2024"""))
cells.append(code('''metrics = ["customers_R2", "customers_MAE", "customers_rho", "duration_R2", "duration_MAE", "duration_rho",
           "severe_ROC_AUC", "severe_PR_AUC", "severe_share"]
blocks, y24 = wf[wf["test_years"] != "2024"], wf[wf["test_years"] == "2024"]
summary = pd.DataFrame({"2008-2023 blocks: mean": blocks[metrics].mean(), "2008-2023 blocks: std": blocks[metrics].std(),
                        "2008-2023 blocks: min": blocks[metrics].min(), "2008-2023 blocks: max": blocks[metrics].max(),
                        "2024": y24[metrics].iloc[0]})
summary.to_csv(os.path.join(OUT_DIR, "exp1_summary.csv"))
display(summary.round(3))'''))

cells.append(md("""### 7.2 Plot: performance per test block"""))
cells.append(code('''fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
x = np.arange(len(wf))
for ax, (col, title) in zip(axes, [("customers_R2", "Customers affected: test R²"), ("duration_R2", "Duration: test R²"),
                                    ("severe_ROC_AUC", "Severe or not: test ROC-AUC")]):
    ax.plot(x[:-1], wf[col].values[:-1], "o-", color="#2a78d6", label="2-year blocks (expanding window)")
    ax.plot(x[-1], wf[col].values[-1], "D", color="#eb6834", markersize=9, label="2024 (separate check)")
    ax.axhline(blocks[col].mean(), color="#8a8984", ls="--", lw=1, label="mean of 2008-2023 blocks")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(wf["test_years"], rotation=45, ha="right")
    ax.grid(axis="y", color="#e6e5e1")
axes[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "exp1_walk_forward.png"), dpi=150)
plt.show()'''))

cells.append(md("""## 8. Is 2024 different? Evidence from the data

Data properties per year (all outages in the strict-rule table). Severe share uses the Q1 / Q3 thresholds of the 2000–2021 training years, so years are comparable."""))
cells.append(code('''def year_profile(df):
    ref = df[df["outage_start"] < pd.Timestamp("2022-01-01")]
    qc = ref["log_customers"].quantile([0.25, 0.75]).values
    qd = ref["log_duration"].quantile([0.25, 0.75]).values
    d = df.assign(year=df["outage_start"].dt.year)
    both = d["log_customers"].notna() & d["log_duration"].notna()
    d["severe"] = np.where(both, ((d["log_customers"] > qc[1]) | (d["log_duration"] > qd[1])).astype(float), np.nan)
    g = d.groupby("year")
    return pd.DataFrame({
        "outages": g.size(),
        "customers known": g["log_customers"].apply(lambda s: s.notna().mean()),
        "duration known": g["log_duration"].apply(lambda s: s.notna().mean()),
        "median customers": g["customers_affected"].median(),
        "median duration (min)": g["outage_duration"].median(),
        "severe share": g["severe"].mean(),
        "share intentional attack": g["cause_reported"].apply(lambda s: (s == "intentional attack").mean()),
        "share severe weather": g["cause_reported"].apply(lambda s: (s == "severe weather").mean()),
        "share with active storm": g["noaa_has_storm"].mean(),
    })


profile = year_profile(tables[WF_RULE])
profile.to_csv(os.path.join(OUT_DIR, "exp1_year_profile.csv"))
display(profile.loc[2014:].round(2))'''))

cells.append(md("""## 9. Notes for interpretation

- Each block's model only sees earlier years, as in real use. Early blocks have less training data.
- PR-AUC depends on the share of severe outages in each block (`severe_share`); ROC-AUC does not, so use ROC-AUC to compare blocks.
- The 2024 check uses the same model and setup as the blocks; section 8 shows how the 2024 data itself differs (reporting, missing values, cause mix)."""))


cells.append(md("""## 10. Does the data really vary over time?

Experiment 1 shows that performance varies between periods. Two questions remain:
1. **Is the variation real, or only noise from small test blocks?** → bootstrap confidence intervals (10.1).
2. **Does the data itself change, not only the model's score?** → target distributions (10.2), cause mix (10.3) and adversarial validation of the input features (10.4)."""))

cells.append(md("""### 10.1 Bootstrap confidence intervals per test block

Each test block is resampled 500 times (with replacement); the 2.5–97.5 percentiles give a 95% interval. Blocks whose intervals do not overlap differ by more than chance."""))
cells.append(code("""rng = np.random.default_rng(SEED)
B = 500
ci_rows = []
for (block, task), (y, p) in WF_PRED.items():
    kind = TASKS[task][0]
    stats = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if kind == "reg":
            stats.append(r2_score(y[i], p[i]))
        elif len(np.unique(y[i])) == 2:
            stats.append(roc_auc_score(y[i], p[i]))
    point = r2_score(y, p) if kind == "reg" else roc_auc_score(y, p)
    ci_rows.append({"test_years": block, "task": task, "metric": "R2" if kind == "reg" else "ROC_AUC", "n": len(y),
                    "value": point, "ci_low": np.percentile(stats, 2.5), "ci_high": np.percentile(stats, 97.5)})
ci = pd.DataFrame(ci_rows)
ci.to_csv(os.path.join(OUT_DIR, "exp1_bootstrap_ci.csv"), index=False)
display(ci.pivot_table(index="test_years", columns="task", values=["value", "ci_low", "ci_high"], sort=False).round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
order = list(wf["test_years"])
for ax, (task, title) in zip(axes, [("customers_reg", "Customers: test R² (95% CI)"), ("duration_reg", "Duration: test R² (95% CI)"),
                                     ("is_severe", "Severe: test ROC-AUC (95% CI)")]):
    c = ci[ci["task"] == task].set_index("test_years").reindex(order)
    x = np.arange(len(c))
    col = ["#eb6834" if b == "2024" else "#2a78d6" for b in c.index]
    ax.errorbar(x, c["value"], yerr=[c["value"] - c["ci_low"], c["ci_high"] - c["value"]], fmt="none", ecolor="#9db8de", lw=2)
    ax.scatter(x, c["value"], color=col, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(c.index, rotation=45, ha="right"); ax.grid(axis="y", color="#e6e5e1")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp1_bootstrap_ci.png"), dpi=150); plt.show()"""))

cells.append(md("""### 10.2 Target distributions by period

Box plots of the two targets (log scale) per period. A shift in the box means the outages themselves changed (not only the model's score)."""))
cells.append(code("""PERIODS = [(2000, 2007)] + WF_BLOCKS + [WF_2024]
def period_label(y):
    for a, b in PERIODS:
        if a <= y <= b:
            return f"{a}-{b}" if b > a else str(a)
dd = tables[WF_RULE].assign(period=tables[WF_RULE]["outage_start"].dt.year.map(period_label))
labels = [f"{a}-{b}" if b > a else str(a) for a, b in PERIODS]
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for ax, (col, title) in zip(axes, [("log_customers", "log(1 + customers per 100k)"), ("log_duration", "log(1 + minutes)")]):
    data_ = [dd.loc[dd["period"] == l, col].dropna().values for l in labels]
    bp = ax.boxplot(data_, labels=labels, showfliers=False, patch_artist=True)
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor("#fde3d6" if labels[k] == "2024" else "#dbe8f8")
    ax.set_title(title, loc="left", fontweight="bold"); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", color="#e6e5e1")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp1_target_distributions.png"), dpi=150); plt.show()
tq = dd.groupby("period", sort=False)[["log_customers", "log_duration"]].median().reindex(labels)
tq.to_csv(os.path.join(OUT_DIR, "exp1_target_medians.csv"))
display(tq.round(2))"""))

cells.append(md("""### 10.3 Cause mix by period

Share of each reported cause per period. Non-weather events (for example intentional attacks) are the hardest to predict from weather data, so a changing mix changes how predictable a period is."""))
cells.append(code("""mix = pd.crosstab(dd["period"], dd["cause_reported"], normalize="index").reindex(labels)
top = mix.mean().sort_values(ascending=False).index[:6]
mix = mix[top].assign(other=1 - mix[top].sum(axis=1))
mix.to_csv(os.path.join(OUT_DIR, "exp1_cause_mix.csv"))
ax = mix.plot(kind="bar", stacked=True, figsize=(12, 4.5), width=0.8,
              color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#c9c9c4"])
ax.set_ylabel("share of outages"); ax.set_xlabel(""); ax.set_title("Reported cause by period", loc="left", fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "exp1_cause_mix.png"), dpi=150); plt.show()
display(mix.round(2))"""))

cells.append(md("""### 10.4 Adversarial validation: do the input features shift?

For each block, a classifier tries to tell the block apart from the **previous** block using only **event-level inputs**: NOAA storm and NWS warning features, time of day / month / weekend, state and region. AUC ≈ 0.5 means the two periods look the same; AUC well above 0.5 means the situations in which outages happen have shifted.
Economy, population and outage-history features are left out on purpose: they are yearly or running averages (for example US GDP per person, global history means), and the El Niño / La Niña `climate_category` is the same for a whole season, so they identify the **calendar year** by construction and would make every comparison trivially AUC = 1. NWS features are left out for blocks that include years before 2006, because they are missing by design there."""))
cells.append(code("""from sklearn.model_selection import StratifiedKFold, cross_val_predict
EVENT_NUM = [c for c in NOAA_NUM + NWS_NUM + ["start_month", "start_hour", "is_weekend", "month_sin", "month_cos", "hour_sin", "hour_cos",
                                              "n_states_affected"] if c in df.columns]
blocks_all = WF_BLOCKS + [WF_2024]
adv_rows, adv_imp = [], {}
for (a0, a1), (b0, b1) in zip(blocks_all[:-1], blocks_all[1:]):
    yr = df["outage_start"].dt.year
    prev, cur = df[(yr >= a0) & (yr <= a1)], df[(yr >= b0) & (yr <= b1)]
    cols_num = [c for c in EVENT_NUM if not (a0 < 2006 and c.startswith("nws_"))]
    cols_cat = [c for c in cat if c != "climate_category"]
    X = pd.concat([prev, cur])[cols_num + cols_cat]
    y = np.r_[np.zeros(len(prev)), np.ones(len(cur))]
    pipe = Pipeline([("pre", build_preprocessor(cols_num, cols_cat)),
                     ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1))])
    p = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    pipe.fit(X, y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    lab_prev = f"{a0}-{a1}" if a1 > a0 else str(a0); lab_cur = f"{b0}-{b1}" if b1 > b0 else str(b0)
    adv_rows.append({"compare": f"{lab_prev} vs {lab_cur}", "n_prev": len(prev), "n_cur": len(cur), "AUC": roc_auc_score(y, p),
                     "top shifted features": ", ".join(i.split("__")[-1] for i in imp.index[:3])})
adv = pd.DataFrame(adv_rows)
adv.to_csv(os.path.join(OUT_DIR, "exp1_adversarial_validation.csv"), index=False)
display(adv.round(3))
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#eb6834" if "2024" in c else "#2a78d6" for c in adv["compare"]]
ax.bar(adv["compare"], adv["AUC"], color=colors)
ax.axhline(0.5, color="#8a8984", ls="--", lw=1); ax.text(len(adv) - 0.5, 0.505, "no shift (0.5)", ha="right", fontsize=8, color="#8a8984")
for i, v in enumerate(adv["AUC"]):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC (previous vs current block)")
ax.set_title("Adversarial validation: input-feature shift between neighbouring periods", loc="left", fontweight="bold")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "exp1_adversarial_validation.png"), dpi=150); plt.show()"""))

cells.append(md("""### 10.5 What section 10 shows

Read together with section 8:
- **Bootstrap (10.1):** blocks whose intervals do not overlap differ by more than chance.
- **Targets and causes (10.2, 10.3):** whether the outages themselves change between periods.
- **Adversarial validation (10.4):** whether the input features shift; AUC well above 0.5 means the model is asked to predict a different mix of situations than it learned from."""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "tf_cpu", "language": "python", "name": "python3"}
json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
open(DST, "a").write("\n")
print("wrote", DST, len(cells), "cells")
