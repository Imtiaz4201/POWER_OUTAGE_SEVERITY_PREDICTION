"""Build final_stage2/exp6_error_diagnosis/exp6_error_diagnosis.ipynb."""
import sys
sys.path.insert(0, "/Users/imtiaz/Documents/po1/final_stage2/build")
from common import md, code, data_cells, write, meta
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp6_error_diagnosis/exp6_error_diagnosis.ipynb"

INTRO = """# Final stage, Experiment 6: error diagnosis

**Problem statement:** *Can we predict how severe a major power outage will become, using historical outage data and recent storm information available when the outage starts?*

**Questions of this experiment:** Experiments 2–5 showed that models, tuning, training years and feature selection do not raise accuracy. This experiment asks **where the models fail and what information is missing**.

**Design (decided before running):**

| | |
|---|---|
| Setup | strict storm rule (R2), main split: train before 2022, test 2022–2023 (2024 excluded) |
| Models | the cross-validation-selected tuned models of Experiment 2 (scenario A and scenario B) |
| Predictions analysed | (1) time-based cross-validation predictions on the training years (4 folds; main evidence, more outages) and (2) test predictions 2022–2023 |
| Where does the model fail? | errors grouped by reported cause, weather vs non-weather cause, and data source. **The cause is used only to group the errors; it is not a model input in scenario A.** |
| What is missing? | (a) scenario A vs scenario B (cause known): how much the cause alone would add; (b) range compression: are large outages under-predicted and small ones over-predicted? |
| Severe detection | recall and false-alarm rate by cause (threshold chosen for F1 on the cross-validation predictions) |"""

cells = [md(INTRO)] + data_cells()

cells.append(md("""## 7. Setup and predictions"""))
cells.append(code('''import ast
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

EXP2 = os.path.join(os.path.dirname(os.getcwd()), "exp2_model_search", "results", "exp2_best_by_cv.csv")
sel = pd.read_csv(EXP2)
sel = sel[(sel["setup"] == "2a_main") & (sel["rule"] == "R2_strict")].set_index(["scenario", "task"])
BASE = {
    ("reg", "Ridge"): Ridge(), ("reg", "Lasso"): Lasso(max_iter=20000),
    ("reg", "RandomForest"): RandomForestRegressor(random_state=SEED, n_jobs=-1),
    ("reg", "XGBoost"): XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0),
    ("bin", "Logistic"): LogisticRegression(class_weight="balanced", max_iter=5000),
    ("bin", "RandomForest"): RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    ("bin", "XGBoost"): XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0, eval_metric="logloss"),
}


def tuned_model(scen, task, y_train):
    kind = TASKS[task][0]
    text = re.sub("np[.][a-z0-9]+[(]([^)]*)[)]", lambda m: m.group(1), sel.loc[(scen, task), "best_params"])
    name = sel.loc[(scen, task), "model"]
    m = clone(BASE[(kind, name)]).set_params(**ast.literal_eval(text))
    if name == "XGBoost" and kind == "bin":
        m.set_params(scale_pos_weight=(1 - y_train.mean()) / max(y_train.mean(), 1e-6))
    return m


display(sel[["model", "cv_mean"]].unstack(0))

RULE = "strict"
SPLIT, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
df = tables[RULE]
TRAIN, TEST = df[df["outage_start"] < SPLIT], df[(df["outage_start"] >= SPLIT) & (df["outage_start"] < END)]
CV = TimeSeriesSplit(n_splits=4)
SCENS = ["A_pre_event", "B_with_cause"]


def collect(scen, task):
    """Cross-validation (training years) and test predictions of the tuned model, with grouping columns."""
    kind, target = TASKS[task]
    num, cat = feature_sets(df, scen)
    d_tr = task_rows(TRAIN, kind, target).sort_values("outage_start", kind="stable")
    d_te = task_rows(TEST, kind, target)
    y_tr, y_te = make_labels(kind, target, TRAIN, d_tr, d_te)
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)

    def fit(d, y):
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("model", tuned_model(scen, task, y))])
        return pipe.fit(d[num + cat], y, model__sample_weight=recency_weights(d))

    oof = np.full(len(y_tr), np.nan)
    for a, b in CV.split(d_tr):
        oof[b] = predict(fit(d_tr.iloc[a], y_tr[a]), kind, d_tr.iloc[b][num + cat])
    p_te = predict(fit(d_tr, y_tr), kind, d_te[num + cat])
    out = []
    for split, d, y, p in [("cv", d_tr, y_tr, oof), ("test", d_te, y_te, p_te)]:
        out.append(pd.DataFrame({"scenario": scen, "task": task, "split": split, "y": y, "p": p,
                                 "cause": d["cause_reported"].values, "source": d["source"].values,
                                 "storm_active": d["noaa_has_storm"].values, "year": d["outage_start"].dt.year.values}))
    return pd.concat(out)[lambda x: x["p"].notna()]


t0 = time.time()
pred = pd.concat([collect(s, t) for s in SCENS for t in TASKS], ignore_index=True)
pred["weather_cause"] = np.where(pred["cause"] == "severe weather", "weather cause", "non-weather cause")
pred.to_csv(os.path.join(OUT_DIR, "exp6_predictions.csv"), index=False)
print(f"predictions collected in {time.time() - t0:.0f}s; rows per split:")
display(pred.groupby(["scenario", "task", "split"]).size().unstack())'''))

cells.append(md("""## 8. Overall scores: scenario A vs scenario B (how much would the cause add?)

Scenario B gives the model the reported cause, which is not known when the outage starts. The gap between A and B is the value of that one missing piece of information."""))
cells.append(code('''rows = []
for (scen, task, split), g in pred.groupby(["scenario", "task", "split"]):
    kind = TASKS[task][0]
    rows.append({"task": task, "split": split, "scenario": scen[0],
                 "score": r2_score(g["y"], g["p"]) if kind == "reg" else roc_auc_score(g["y"], g["p"])})
ov = pd.DataFrame(rows).pivot(index=["task", "split"], columns="scenario", values="score")
ov["gain from knowing the cause (B − A)"] = ov["B"] - ov["A"]
ov.to_csv(os.path.join(OUT_DIR, "exp6_overall_A_vs_B.csv"))
print("R² for customers and duration, ROC-AUC for is_severe")
display(ov.round(3))'''))

cells.append(md("""## 9. Where do the regression models fail? Errors by cause (scenario A)

For each cause: share of outages, mean absolute error (MAE), bias (mean prediction − mean actual; positive = over-prediction) and **share of the total squared error**. A cause whose share of error is much larger than its share of outages is where the model fails. Causes with fewer than 20 outages are grouped as "other"."""))
cells.append(code('''def by_group(g, col, min_n=20):
    g = g.copy()
    counts = g[col].value_counts()
    g[col] = g[col].where(g[col].map(counts) >= min_n, "other")
    g["sq"] = (g["p"] - g["y"]) ** 2
    t = g.groupby(col).apply(lambda x: pd.Series({
        "n": len(x), "share of outages": len(x) / len(g), "mean actual": x["y"].mean(), "mean predicted": x["p"].mean(),
        "bias": (x["p"] - x["y"]).mean(), "MAE": (x["p"] - x["y"]).abs().mean(),
        "share of squared error": x["sq"].sum() / g["sq"].sum()}))
    t["error share / outage share"] = t["share of squared error"] / t["share of outages"]
    return t.sort_values("n", ascending=False)


A = pred[pred["scenario"] == "A_pre_event"]
cause_tables = {}
for task in ["customers_reg", "duration_reg"]:
    for split in ["cv", "test"]:
        t = by_group(A[(A["task"] == task) & (A["split"] == split)], "cause")
        cause_tables[(task, split)] = t
        t.to_csv(os.path.join(OUT_DIR, f"exp6_error_by_cause_{task}_{split}.csv"))
        print(f"{task}, {split} (log scale)"); display(t.round(3))'''))

cells.append(md("""### 9.1 Plot: share of outages vs share of squared error, by cause (cross-validation, scenario A)"""))
cells.append(code('''fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, task in zip(axes, ["customers_reg", "duration_reg"]):
    t = cause_tables[(task, "cv")]
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["share of outages"], 0.4, color="#8a8984", label="share of outages")
    ax.barh(y - 0.2, t["share of squared error"], 0.4, color="#eb6834", label="share of squared error")
    ax.set_yticks(y); ax.set_yticklabels(t.index); ax.invert_yaxis()
    ax.set_title(f"{task}: where the error comes from", loc="left", fontweight="bold"); ax.grid(axis="x", color="#e6e5e1")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp6_error_by_cause.png"), dpi=150); plt.show()'''))

cells.append(md("""### 9.2 Weather vs non-weather cause, and data source (scenario A)

R² **within** each group shows how well the model separates outages of the same kind."""))
cells.append(code('''rows = []
for task in ["customers_reg", "duration_reg"]:
    for split in ["cv", "test"]:
        g = A[(A["task"] == task) & (A["split"] == split)]
        for col in ["weather_cause", "source"]:
            for name, x in g.groupby(col):
                rows.append({"task": task, "split": split, "group by": col, "group": name, "n": len(x),
                             "R² within group": r2_score(x["y"], x["p"]), "MAE": mean_absolute_error(x["y"], x["p"]),
                             "bias": (x["p"] - x["y"]).mean(),
                             "share of squared error": ((x["p"] - x["y"]) ** 2).sum() / ((g["p"] - g["y"]) ** 2).sum()})
        rows.append({"task": task, "split": split, "group by": "all", "group": "all outages", "n": len(g),
                     "R² within group": r2_score(g["y"], g["p"]), "MAE": mean_absolute_error(g["y"], g["p"]),
                     "bias": (g["p"] - g["y"]).mean(), "share of squared error": 1.0})
grp = pd.DataFrame(rows)
grp.to_csv(os.path.join(OUT_DIR, "exp6_error_by_weather_and_source.csv"), index=False)
display(grp.round(3))'''))

cells.append(md("""## 10. What is missing? Range compression

Outages are sorted into five equal groups by their **actual** value (smallest to largest). If the model had the information to tell small and large outages apart, the mean prediction would follow the mean actual value. A flat prediction line means the model predicts close to the average for everything: it lacks information about the **size** of the event."""))
cells.append(code('''rows = []
for (scen, task), g in pred[(pred["split"] == "cv") & pred["task"].isin(["customers_reg", "duration_reg"])].groupby(["scenario", "task"]):
    g = g.assign(bin=pd.qcut(g["y"], 5, labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "highest 20%"]))
    for b, x in g.groupby("bin", observed=True):
        rows.append({"scenario": scen[0], "task": task, "actual-value group": b, "mean actual": x["y"].mean(),
                     "mean predicted": x["p"].mean(), "bias": (x["p"] - x["y"]).mean()})
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(OUT_DIR, "exp6_range_compression.csv"), index=False)
display(comp[comp["scenario"] == "A"].round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, task in zip(axes, ["customers_reg", "duration_reg"]):
    a = comp[(comp["task"] == task) & (comp["scenario"] == "A")]; b = comp[(comp["task"] == task) & (comp["scenario"] == "B")]
    x = np.arange(5)
    ax.plot(x, a["mean actual"], "o-", color="#0b0b0b", label="actual")
    ax.plot(x, a["mean predicted"], "s-", color="#2a78d6", label="predicted, scenario A (pre-event)")
    ax.plot(x, b["mean predicted"], "^--", color="#eb6834", label="predicted, scenario B (cause known)")
    ax.set_xticks(x); ax.set_xticklabels(a["actual-value group"], fontsize=8)
    ax.set_title(f"{task}: mean actual vs mean predicted (log scale, CV)", loc="left", fontweight="bold", fontsize=10)
    ax.grid(color="#e6e5e1")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp6_range_compression.png"), dpi=150); plt.show()

spread = pred[(pred["split"] == "cv") & pred["task"].isin(["customers_reg", "duration_reg"])].groupby(["task", "scenario"]).apply(
    lambda x: pd.Series({"std of actual": x["y"].std(), "std of predicted": x["p"].std(), "ratio": x["p"].std() / x["y"].std()}))
spread.to_csv(os.path.join(OUT_DIR, "exp6_prediction_spread.csv"))
print("Spread of predictions compared with the spread of actual values (1.0 = same spread)")
display(spread.round(2))'''))

cells.append(md("""## 11. Severe detection by cause (scenario A)

Threshold chosen for F1 on the cross-validation predictions, then applied to both splits. Recall = share of truly severe outages that were flagged; false-alarm rate = share of non-severe outages that were flagged."""))
cells.append(code('''S = A[A["task"] == "is_severe"]
cvp = S[S["split"] == "cv"]
grid = np.linspace(0.05, 0.95, 91)
THR = grid[int(np.argmax([f1_score(cvp["y"], cvp["p"] >= t, zero_division=0) for t in grid]))]
rows = []
for split in ["cv", "test"]:
    g = S[S["split"] == split].copy()
    counts = g["cause"].value_counts()
    g["cause"] = g["cause"].where(g["cause"].map(counts) >= 15, "other")
    for name, x in list(g.groupby("cause")) + [("all outages", g)]:
        flag = x["p"] >= THR
        sev = x["y"] == 1
        rows.append({"split": split, "cause": name, "n": len(x), "severe share": sev.mean(), "mean predicted probability": x["p"].mean(),
                     "recall": flag[sev].mean() if sev.any() else np.nan,
                     "false-alarm rate": flag[~sev].mean() if (~sev).any() else np.nan,
                     "ROC-AUC within group": roc_auc_score(x["y"], x["p"]) if x["y"].nunique() == 2 else np.nan})
sevt = pd.DataFrame(rows)
sevt.to_csv(os.path.join(OUT_DIR, "exp6_severe_by_cause.csv"), index=False)
print(f"decision threshold = {THR:.2f}")
display(sevt.round(3))'''))

cells.append(md("""## 12. Notes for interpretation

- **Cause is only a grouping variable here.** In scenario A the model never sees it; the tables show how the errors are distributed over outage types that the model cannot tell apart at outage start.
- **Error share / outage share > 1** means that cause contributes more error than its share of outages: these are the outages the model fails on.
- **Scenario B − A** measures the value of knowing the cause. What remains unexplained even in scenario B is information that none of our data contains (for example the local network that is hit and the number of customers it serves).
- **Range compression:** a prediction spread much smaller than the actual spread means the features cannot tell small from large events.
- Cross-validation predictions cover the training years (first fold block excluded); test predictions cover 2022–2023."""))

write(cells, DST, meta())
