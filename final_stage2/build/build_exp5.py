"""Build final_stage2/exp5_feature_selection/exp5_feature_selection.ipynb."""
import sys
sys.path.insert(0, "/Users/imtiaz/Documents/po1/final_stage2/build")
from common import md, code, data_cells, write, meta
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp5_feature_selection/exp5_feature_selection.ipynb"

INTRO = """# Final stage, Experiment 5: feature selection and feature importance

**Problem statement:** *Can we predict how severe a major power outage will become, using historical outage data and recent storm information available when the outage starts?*

**Questions of this experiment:**
1. Do the models work as well (or better) with **fewer features**?
2. **Which features matter most** for each target?

**Design (decided before running):**

| | |
|---|---|
| Setup | strict storm rule (R2), scenario A (pre-event), main split: train before 2022, test 2022–2023 (2024 excluded) |
| Models | the cross-validation-selected tuned model of Experiment 2 for each target (same settings) |
| Feature ranking | random-forest importance, computed **inside each cross-validation fold on that fold's training part only**, so the validation data never influences which features are chosen. One-hot and missing-value columns are summed back to their original feature. |
| Sizes | top 40, 50, 60, 80, 100 features, and all 115 for reference |
| Score | time-based cross-validation on the training years (4 folds): mean and standard deviation per size |
| Choice | the size with the best cross-validation mean (ties → smaller size) |
| Test | the chosen size is retrained on all training years (ranking also from all training years) and scored **once** on 2022–2023 |
| Metrics | R² for customers and duration, PR-AUC for is_severe in cross-validation (as in Experiment 2); ROC-AUC also shown on test |"""

cells = [md(INTRO)] + data_cells()

cells.append(md("""## 7. Setup: models from Experiment 2, feature ranking"""))
cells.append(code('''import ast
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

EXP2 = os.path.join(os.path.dirname(os.getcwd()), "exp2_model_search", "results", "exp2_best_by_cv.csv")
sel = pd.read_csv(EXP2)
sel = sel[(sel["setup"] == "2a_main") & (sel["scenario"] == "A_pre_event") & (sel["rule"] == "R2_strict")].set_index("task")
BASE = {
    ("reg", "Ridge"): Ridge(), ("reg", "Lasso"): Lasso(max_iter=20000),
    ("reg", "RandomForest"): RandomForestRegressor(random_state=SEED, n_jobs=-1),
    ("reg", "XGBoost"): XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0),
    ("bin", "Logistic"): LogisticRegression(class_weight="balanced", max_iter=5000),
    ("bin", "RandomForest"): RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    ("bin", "XGBoost"): XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0, eval_metric="logloss"),
}


def tuned_model(task, y_train=None):
    kind = TASKS[task][0]
    text = re.sub("np[.][a-z0-9]+[(]([^)]*)[)]", lambda m: m.group(1), sel.loc[task, "best_params"])
    name, params = sel.loc[task, "model"], ast.literal_eval(text)
    m = clone(BASE[(kind, name)]).set_params(**params)
    if name == "XGBoost" and kind == "bin" and y_train is not None:
        m.set_params(scale_pos_weight=(1 - y_train.mean()) / max(y_train.mean(), 1e-6))
    return m


display(sel[["model", "cv_mean", "best_params"]])

RULE, SCEN = "strict", "A_pre_event"
SPLIT, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
df = tables[RULE]
TRAIN, TEST = df[df["outage_start"] < SPLIT], df[(df["outage_start"] >= SPLIT) & (df["outage_start"] < END)]
NUM, CAT = feature_sets(df, SCEN)
FEATS = NUM + CAT
SIZES = [40, 50, 60, 80, 100, len(FEATS)]
CV = TimeSeriesSplit(n_splits=4)
GROUP_OF = {}
for gname, cols in {"economy / population / land": STATIC_NUM, "time": TIME_NUM + ["regime_2024"], "NOAA storms": NOAA_NUM,
                    "NWS warnings": NWS_NUM, "outage history": HIST_NUM + RECENT_HIST + STORM_TE, "area history": AREA_NUM,
                    "region / state / storm type": BASE_CAT}.items():
    for c in cols:
        GROUP_OF[c] = gname
print(f"{len(FEATS)} features ({len(NUM)} numeric + {len(CAT)} categorical); sizes tested: {SIZES}")


def task_data(task):
    kind, target = TASKS[task]
    d_tr = task_rows(TRAIN, kind, target).sort_values("outage_start", kind="stable")
    d_te = task_rows(TEST, kind, target)
    y_tr, y_te = make_labels(kind, target, TRAIN, d_tr, d_te)
    return kind, d_tr, d_te, np.asarray(y_tr), np.asarray(y_te)


def rank_features(kind, d, y):
    """Random-forest importance summed back to the original features (fitted on the given rows only)."""
    pre = build_preprocessor(NUM, CAT).fit(d[FEATS])
    X = pre.transform(d[FEATS])
    rf = (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.33, random_state=SEED, n_jobs=-1) if kind == "reg"
          else RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.33, class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=-1))
    rf.fit(X, y, sample_weight=recency_weights(d))
    imp = pd.Series(0.0, index=FEATS)
    for name, v in zip(pre.get_feature_names_out(), rf.feature_importances_):
        part, col = name.split("__", 1)
        if part == "num":
            imp[col.replace("missingindicator_", "")] += v
        else:
            imp[next(c for c in CAT if col.startswith(c + "_") or col == c)] += v
    return imp.sort_values(ascending=False)


def metric(kind, y, p):
    return r2_score(y, p) if kind == "reg" else average_precision_score(y, p)


def fit_predict(task, kind, d_fit, y_fit, d_eval, cols):
    num, cat = [c for c in NUM if c in cols], [c for c in CAT if c in cols]
    pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("model", tuned_model(task, y_fit))])
    pipe.fit(d_fit[num + cat], y_fit, model__sample_weight=recency_weights(d_fit))
    return predict(pipe, kind, d_eval[num + cat])'''))

cells.append(md("""## 8. Cross-validation for each number of features

In each fold: rank the features on the fold's training part, keep the top k, fit the tuned model, score the validation part."""))
cells.append(code('''cv_rows = []
t0 = time.time()
for task in TASKS:
    kind, d_tr, d_te, y_tr, y_te = task_data(task)
    for f_i, (a, b) in enumerate(CV.split(d_tr)):
        ranking = rank_features(kind, d_tr.iloc[a], y_tr[a])
        for k in SIZES:
            p = fit_predict(task, kind, d_tr.iloc[a], y_tr[a], d_tr.iloc[b], set(ranking.index[:k]))
            cv_rows.append({"task": task, "k": k, "fold": f_i, "score": metric(kind, y_tr[b], p)})
    print(f"{task} done ({time.time() - t0:.0f}s)", flush=True)
cvf = pd.DataFrame(cv_rows)
cvs = cvf.groupby(["task", "k"])["score"].agg(cv_mean="mean", cv_std="std").reset_index()
best_mean = cvs.groupby("task")["cv_mean"].transform("max")
cvs["chosen"] = False
for task, g in cvs.groupby("task"):
    cvs.loc[g[g["cv_mean"] == g["cv_mean"].max()].sort_values("k").index[0], "chosen"] = True
cvf.to_csv(os.path.join(OUT_DIR, "exp5_cv_folds.csv"), index=False)
cvs.to_csv(os.path.join(OUT_DIR, "exp5_cv_summary.csv"), index=False)
print("score = R² (customers, duration) or PR-AUC (is_severe)")
display(cvs.pivot(index="k", columns="task", values="cv_mean").round(3))
display(cvs.pivot(index="k", columns="task", values="cv_std").round(3).add_suffix(" (std)"))
display(cvs[cvs["chosen"]][["task", "k", "cv_mean", "cv_std"]].round(3))'''))

cells.append(md("""### 8.1 Plot: cross-validation score vs number of features (mean ± std)"""))
cells.append(code('''fig, axes = plt.subplots(1, 3, figsize=(16, 4))
names = {"customers_reg": "Customers: CV R²", "duration_reg": "Duration: CV R²", "is_severe": "Severe: CV PR-AUC"}
for ax, task in zip(axes, TASKS):
    s = cvs[cvs["task"] == task]
    ax.errorbar(s["k"], s["cv_mean"], yerr=s["cv_std"], fmt="o-", color="#2a78d6", ecolor="#9db8de", capsize=3)
    c = s[s["chosen"]]
    ax.scatter(c["k"], c["cv_mean"], s=120, facecolors="none", edgecolors="#eb6834", linewidths=2, zorder=3, label="chosen")
    ax.set_title(names[task], loc="left", fontweight="bold"); ax.set_xlabel("number of features"); ax.grid(color="#e6e5e1")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp5_cv_vs_k.png"), dpi=150); plt.show()'''))

cells.append(md("""## 9. Test the chosen number of features once (2022–2023)

Ranking and model are fitted on all training years. "All features" (the Experiment 2 setup) is shown alongside for comparison."""))
cells.append(code('''test_rows, final_rank = [], {}
for task in TASKS:
    kind, d_tr, d_te, y_tr, y_te = task_data(task)
    ranking = rank_features(kind, d_tr, y_tr)
    final_rank[task] = ranking
    k_star = int(cvs[(cvs["task"] == task) & cvs["chosen"]]["k"].iloc[0])
    for k in dict.fromkeys([k_star, len(FEATS)]):
        p = fit_predict(task, kind, d_tr, y_tr, d_te, set(ranking.index[:k]))
        row = {"task": task, "k": k, "role": ("chosen by CV" if k == k_star else "") + (" / all features" if k == len(FEATS) else "")}
        if kind == "reg":
            row.update(test_R2=r2_score(y_te, p), test_MAE=mean_absolute_error(y_te, p), test_rho=spearmanr(y_te, p).correlation)
        else:
            row.update(test_ROC_AUC=roc_auc_score(y_te, p), test_PR_AUC=average_precision_score(y_te, p), test_base_rate=y_te.mean())
        test_rows.append(row)
tst = pd.DataFrame(test_rows)
tst.to_csv(os.path.join(OUT_DIR, "exp5_test.csv"), index=False)
display(tst.round(3))'''))

cells.append(md("""## 10. Feature importance (all training years)

Top 20 features per target by random-forest importance (summed over one-hot and missing-value columns), coloured by feature group, and the share of total importance per group."""))
cells.append(code('''GCOL = {"economy / population / land": "#8a8984", "time": "#eda100", "NOAA storms": "#2a78d6", "NWS warnings": "#1baf7a",
        "outage history": "#eb6834", "area history": "#e87ba4", "region / state / storm type": "#6250d6"}
imp_rows = []
for task, r in final_rank.items():
    for rank, (f, v) in enumerate(r.items(), 1):
        imp_rows.append({"task": task, "rank": rank, "feature": f, "group": GROUP_OF.get(f, "other"), "importance": v})
imp = pd.DataFrame(imp_rows)
imp.to_csv(os.path.join(OUT_DIR, "exp5_feature_importance.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax, task in zip(axes, TASKS):
    s = imp[(imp["task"] == task) & (imp["rank"] <= 20)].iloc[::-1]
    ax.barh(s["feature"], s["importance"], color=[GCOL.get(g, "#c9c9c4") for g in s["group"]])
    ax.set_title(f"{task}: top 20 features", loc="left", fontweight="bold"); ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color="#e6e5e1")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GCOL.values()]
fig.legend(handles, GCOL.keys(), loc="lower center", ncol=7, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT_DIR, "exp5_top20_features.png"), dpi=150); plt.show()

share = imp.groupby(["task", "group"])["importance"].sum().unstack(0)
share = (share / share.sum()).round(3)
share.to_csv(os.path.join(OUT_DIR, "exp5_importance_by_group.csv"))
print("Share of total importance per feature group")
display(share)'''))

cells.append(md("""## 11. Notes for interpretation

- Features are ranked inside each fold using only that fold's training rows, so the cross-validation scores are not biased by the selection.
- If the scores for 40–100 features are within the fold spread (`cv_std`) of "all features", a smaller model is **equally good** (simpler, easier to explain), not better.
- Random-forest importance shows how much a model *uses* a feature; it is not causal, and correlated features share importance (for example several NWS counts).
- The test period is used once, for the chosen size; "all features" is shown for comparison."""))

write(cells, DST, meta())
