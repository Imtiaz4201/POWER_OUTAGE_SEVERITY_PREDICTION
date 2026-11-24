"""Build final_stage2/exp3_learning_curve_ablation/exp3_learning_curve_ablation.ipynb."""
import sys
sys.path.insert(0, "/Users/imtiaz/Documents/po1/final_stage2/build")
from common import md, code, data_cells, write, meta
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp3_learning_curve_ablation/exp3_learning_curve_ablation.ipynb"

INTRO = """# Final stage, Experiment 3: learning curve and feature-group ablation

**Problem statement:** *Can we predict how severe a major power outage will become, using historical outage data and recent storm information available when the outage starts?*

**Questions of this experiment:**
1. **Is the limit the amount of data?** If accuracy keeps rising with more training outages, more data would help; if it flattens, the limit is the information in the features (3a).
2. **Does older data help, or only recent years?** (3b)
3. **Which information matters?** Remove one feature group at a time and see how much accuracy drops (3c).

**Design (decided before running):**

| | |
|---|---|
| Setup | strict storm rule (R2), scenario A (pre-event), main split: train before 2022, test 2022–2023 (2024 excluded) |
| Model | one fixed random forest per target (same settings as Experiment 1), so only the data changes |
| 3a Learning curve | 10 %, 20 %, …, 100 % of the training outages, 5 random repeats each; scored on the test period |
| 3b History length | train on the last 4, 8, 12, 16 or 22 years before 2022; scored on the test period |
| 3c Ablation | drop one feature group at a time; scored by time-based cross-validation on the training years (4 folds) **and** on the test period |
| Metrics | R² for customers and duration (log scale), ROC-AUC for is_severe |

The data pipeline (sections 0–6) is the same as in the progress-stage notebook."""

cells = [md(INTRO)] + data_cells()

cells.append(md("""## 7. Common setup"""))
cells.append(code('''from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

RULE, SCEN = "strict", "A_pre_event"
SPLIT, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
df = tables[RULE]
TRAIN, TEST = df[df["outage_start"] < SPLIT], df[(df["outage_start"] >= SPLIT) & (df["outage_start"] < END)]
NUM, CAT = feature_sets(df, SCEN)
CV = TimeSeriesSplit(n_splits=4)
METRIC = {"customers_reg": "R²", "duration_reg": "R²", "is_severe": "ROC-AUC"}


def fixed_model(kind):
    if kind == "reg":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.33, random_state=SEED, n_jobs=-1)
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.33,
                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def task_data(task, train=TRAIN, test=TEST):
    kind, target = TASKS[task]
    d_tr = task_rows(train, kind, target).sort_values("outage_start", kind="stable")
    d_te = task_rows(test, kind, target)
    y_tr, y_te = make_labels(kind, target, train, d_tr, d_te)
    return kind, d_tr, d_te, np.asarray(y_tr), np.asarray(y_te)


def score(kind, y, p):
    return r2_score(y, p) if kind == "reg" else roc_auc_score(y, p)


def fit_score(kind, d_tr, y_tr, d_te, y_te, num, cat, rows=None):
    """Fit on (a subset of) the training rows, score on the test rows."""
    if rows is not None:
        d_tr, y_tr = d_tr.iloc[rows], y_tr[rows]
    pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("model", fixed_model(kind))])
    pipe.fit(d_tr[num + cat], y_tr, model__sample_weight=recency_weights(d_tr))
    return score(kind, y_te, predict(pipe, kind, d_te[num + cat]))


def cv_score(kind, d_tr, y_tr, num, cat):
    """Mean score over time-based folds on the training years."""
    s = []
    for a, b in CV.split(d_tr):
        if kind == "bin" and len(np.unique(y_tr[b])) < 2:
            continue
        s.append(fit_score(kind, d_tr.iloc[a], y_tr[a], d_tr.iloc[b], y_tr[b], num, cat))
    return np.mean(s)


print(f"train {len(TRAIN)} outages (before 2022), test {len(TEST)} outages (2022-2023); {len(NUM)} numeric + {len(CAT)} categorical features")'''))

cells.append(md("""## 8. Experiment 3a: learning curve

Random subsets of the training outages (keeping all years), 5 repeats per size. If the curve is still rising at 100 %, more data would likely help."""))
cells.append(code('''rng = np.random.default_rng(SEED)
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lc_rows = []
t0 = time.time()
for task in TASKS:
    kind, d_tr, d_te, y_tr, y_te = task_data(task)
    for f in FRACS:
        for rep in range(1 if f == 1.0 else 5):
            n = int(round(f * len(d_tr)))
            rows = np.sort(rng.choice(len(d_tr), n, replace=False))
            if kind == "bin" and len(np.unique(y_tr[rows])) < 2:
                continue
            lc_rows.append({"task": task, "fraction": f, "n_train": n, "repeat": rep,
                            "test_score": fit_score(kind, d_tr, y_tr, d_te, y_te, NUM, CAT, rows)})
lc = pd.DataFrame(lc_rows)
lc.to_csv(os.path.join(OUT_DIR, "exp3a_learning_curve.csv"), index=False)
lcs = lc.groupby(["task", "fraction"]).agg(n_train=("n_train", "mean"), mean=("test_score", "mean"), std=("test_score", "std")).reset_index()
print(f"learning curve done in {time.time() - t0:.0f}s")
display(lcs.pivot(index="fraction", columns="task", values="mean").round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, task in zip(axes, TASKS):
    s = lcs[lcs["task"] == task]
    ax.plot(s["n_train"], s["mean"], "o-", color="#2a78d6")
    ax.fill_between(s["n_train"], s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), color="#2a78d6", alpha=0.15)
    ax.set_title(f"{task}: test {METRIC[task]}", loc="left", fontweight="bold"); ax.set_xlabel("training outages")
    ax.grid(color="#e6e5e1")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp3a_learning_curve.png"), dpi=150); plt.show()'''))

cells.append(md("""## 9. Experiment 3b: how many years of history?

Train only on the most recent years before 2022; the test period stays 2022–2023. History *features* still use all earlier outages; only the *training rows* change."""))
cells.append(code('''START_YEARS = [2018, 2014, 2010, 2006, 2000]   # last 4, 8, 12, 16, 22 years
hw_rows = []
for task in TASKS:
    for y0 in START_YEARS:
        tr = TRAIN[TRAIN["outage_start"].dt.year >= y0]
        kind, d_tr, d_te, y_tr, y_te = task_data(task, train=tr)
        hw_rows.append({"task": task, "train_years": f"{y0}-2021", "n_years": 2022 - y0, "n_train": len(d_tr),
                        "test_score": fit_score(kind, d_tr, y_tr, d_te, y_te, NUM, CAT)})
hw = pd.DataFrame(hw_rows)
hw.to_csv(os.path.join(OUT_DIR, "exp3b_history_length.csv"), index=False)
display(hw.pivot(index=["train_years", "n_years"], columns="task", values="test_score").round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, task in zip(axes, TASKS):
    s = hw[hw["task"] == task]
    ax.bar(s["train_years"], s["test_score"], color="#2a78d6")
    for i, v in enumerate(s["test_score"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{task}: test {METRIC[task]}", loc="left", fontweight="bold"); ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", color="#e6e5e1")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp3b_history_length.png"), dpi=150); plt.show()'''))

cells.append(md("""## 10. Experiment 3c: feature-group ablation

Each feature group is removed in turn; the change from the full model shows how much that information contributes. A negative change means the group helps. Cross-validation (training years) is the main evidence; test (2022–2023) is shown alongside."""))
cells.append(code('''GROUPS = {
    "Economy, population, land use": STATIC_NUM,
    "Time (month, hour, weekend, regime)": TIME_NUM + ["regime_2024"],
    "NOAA storms": NOAA_NUM,
    "NWS warnings": NWS_NUM,
    "Outage history (state, global, storm status)": HIST_NUM + RECENT_HIST + STORM_TE,
    "Area history (county)": AREA_NUM,
    "Region / state / storm type (categorical)": BASE_CAT,
}
covered = {c for g in GROUPS.values() for c in g}
print("features not in any group (kept in every run):", [c for c in NUM + CAT if c not in covered])
ab_rows = []
t0 = time.time()
for task in TASKS:
    kind, d_tr, d_te, y_tr, y_te = task_data(task)
    full_cv, full_test = cv_score(kind, d_tr, y_tr, NUM, CAT), fit_score(kind, d_tr, y_tr, d_te, y_te, NUM, CAT)
    ab_rows.append({"task": task, "removed": "(none: full model)", "n_features": len(NUM) + len(CAT), "cv": full_cv, "test": full_test})
    for gname, cols in GROUPS.items():
        num = [c for c in NUM if c not in cols]; cat = [c for c in CAT if c not in cols]
        ab_rows.append({"task": task, "removed": gname, "n_features": len(num) + len(cat),
                        "cv": cv_score(kind, d_tr, y_tr, num, cat), "test": fit_score(kind, d_tr, y_tr, d_te, y_te, num, cat)})
    print(f"{task} done", flush=True)
ab = pd.DataFrame(ab_rows)
full = ab[ab["removed"] == "(none: full model)"].set_index("task")
ab["cv_change"] = ab["cv"] - ab["task"].map(full["cv"])
ab["test_change"] = ab["test"] - ab["task"].map(full["test"])
ab.to_csv(os.path.join(OUT_DIR, "exp3c_ablation.csv"), index=False)
print(f"ablation done in {time.time() - t0:.0f}s")
for task in TASKS:
    print(f"{task} ({METRIC[task]})"); display(ab[ab["task"] == task].drop(columns="task").round(3).reset_index(drop=True))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, task in zip(axes, TASKS):
    s = ab[(ab["task"] == task) & (ab["removed"] != "(none: full model)")]
    y = np.arange(len(s))
    ax.barh(y + 0.2, s["cv_change"], 0.4, color="#2a78d6", label="CV (training years)")
    ax.barh(y - 0.2, s["test_change"], 0.4, color="#eb6834", label="test 2022-2023")
    ax.axvline(0, color="#8a8984", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(s["removed"], fontsize=8); ax.invert_yaxis()
    ax.set_title(f"{task}: change in {METRIC[task]} when removed", loc="left", fontweight="bold", fontsize=9)
    ax.grid(axis="x", color="#e6e5e1")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp3c_ablation.png"), dpi=150); plt.show()'''))

cells.append(md("""## 11. Notes for interpretation

- **Learning curve:** a curve that is still rising at 100 % means more data would help; a flat curve means the limit is the information in the features.
- **History length:** better scores with fewer, recent years would mean older data is less relevant (the data changes over time, see Experiment 1).
- **Ablation:** a group whose removal lowers the score carries useful information; a group whose removal does not change the score is redundant with the others (or not useful). Changes of about ±0.01 are within noise.
- All scores use one fixed model (random forest, as in Experiment 1), so they compare *data*, not models; Experiment 2 compares models."""))

write(cells, DST, meta())
