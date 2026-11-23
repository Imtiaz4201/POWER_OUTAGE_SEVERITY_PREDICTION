"""Build final_stage2/exp2_model_search/exp2_model_search.ipynb."""
import sys
sys.path.insert(0, "/Users/imtiaz/Documents/po1/final_stage2/build")
from common import md, code, data_cells, write, meta
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp2_model_search/exp2_model_search.ipynb"

INTRO = """# Final stage, Experiment 2: model comparison with random search

**Problem statement:** *Can we predict how severe a major power outage will become, using historical outage data and recent storm information available when the outage starts?*

**Question of this experiment:** which model works best for each target, storm rule and scenario, and how much does tuning help?

**Design (decided before running):**

| | |
|---|---|
| Setups | **2a main:** train before 2022, test 2022–2023 (2024 excluded, see Experiment 1). **2b source-consistent:** OE-417 reports only, train Jul 2016–2020, test 2021–2023. |
| Runs | **strict storm rule (R2, operational)**, base model (no infrastructure data), scenarios **A (pre-event)** and **B (with cause)**, 3 targets. The four storm rules were compared in the progress stage and performed alike, so the final experiments use the operational rule. |
| Models | regression: Ridge, Lasso, Random Forest, XGBoost; classification (is_severe): Logistic regression, Random Forest, XGBoost |
| Search | `RandomizedSearchCV`, 25 settings per model (fewer when the space is smaller), 4 time-based inner folds on the training years, recency weights |
| Selection | the best model per combination is chosen by **cross-validation score only**; test scores are shown for every model for transparency |
| is_severe | threshold chosen for F1 on the training years' cross-validation predictions |

The data pipeline (sections 0–6) is the same as in the progress-stage notebook. Results are saved after every combination (`results/exp2_all_models.csv`), so the run can be resumed."""

cells = [md(INTRO)] + data_cells()

cells.append(md("""## 7. Setups, models and search spaces"""))
cells.append(code('''import warnings
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings("ignore")

RULES = {"R2_strict": "strict"}  # operational rule (rules R0-R3 were compared in the progress stage)
SCENARIO_LIST = ["A_pre_event", "B_with_cause"]
N_ITER, N_SPLITS = 25, 4
INNER_CV = TimeSeriesSplit(n_splits=N_SPLITS)


def setup_masks(df, setup):
    t = df["outage_start"]
    if setup == "2a_main":
        return t < pd.Timestamp("2022-01-01"), (t >= pd.Timestamp("2022-01-01")) & (t < pd.Timestamp("2024-01-01"))
    oe = df["source"] == "OE-417"            # 2b: OE-417 reports only (one reporting source)
    return oe & (t < pd.Timestamp("2021-01-01")), oe & (t >= pd.Timestamp("2021-01-01")) & (t < pd.Timestamp("2024-01-01"))


SETUPS = ["2a_main", "2b_oe417_only"]

REG_MODELS = {
    "Ridge": (Ridge(), {"alpha": loguniform(1e-2, 1e3)}),
    "Lasso": (Lasso(max_iter=20000), {"alpha": loguniform(1e-4, 1.0)}),
    "RandomForest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                     {"n_estimators": [200, 400], "max_depth": [None, 6, 10, 16], "min_samples_leaf": [1, 3, 5, 10, 20],
                      "max_features": [0.2, 0.33, 0.5, "sqrt"]}),
    "XGBoost": (XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0),
                {"n_estimators": randint(100, 600), "learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 7),
                 "min_child_weight": randint(1, 10), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.4, 0.6),
                 "reg_lambda": loguniform(0.1, 10)}),
}
CLF_MODELS = {
    "Logistic": (LogisticRegression(class_weight="balanced", max_iter=5000), {"C": loguniform(1e-3, 1e2)}),
    "RandomForest": (RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
                     {"n_estimators": [200, 400], "max_depth": [None, 6, 10, 16], "min_samples_leaf": [1, 3, 5, 10, 20],
                      "max_features": [0.2, 0.33, 0.5, "sqrt"]}),
    "XGBoost": (XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0, eval_metric="logloss"),
                {"n_estimators": randint(100, 600), "learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 7),
                 "min_child_weight": randint(1, 10), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.4, 0.6),
                 "reg_lambda": loguniform(0.1, 10)}),
}
print("combinations:", len(SETUPS) * len(RULES) * len(SCENARIO_LIST) * len(TASKS))'''))

cells.append(md("""## 8. Run the search

For every setup × rule × scenario × target, every model is tuned with the random search on the training years. The tuned model is then refit on all training rows and scored on the test period. One row per model is saved immediately."""))
cells.append(code('''def f1_threshold(p, y):
    grid = np.linspace(0.05, 0.95, 91)
    return grid[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid]))]


def oof_predictions(pipe, X, y, w, kind):
    """Time-based out-of-fold predictions on the training years (first block has none)."""
    p = np.full(len(y), np.nan)
    for a, b in INNER_CV.split(X):
        m = clone(pipe).fit(X.iloc[a], y[a], model__sample_weight=w[a])
        p[b] = m.predict(X.iloc[b]) if kind == "reg" else m.predict_proba(X.iloc[b])[:, 1]
    return p


def run_one(setup, rule, scen, task):
    df = tables[RULES[rule]]
    tr_mask, te_mask = setup_masks(df, setup)
    train, test = df[tr_mask], df[te_mask]
    kind, target = TASKS[task]
    num, cat = feature_sets(df, scen)
    d_tr = task_rows(train, kind, target).sort_values("outage_start", kind="stable")
    d_te = task_rows(test, kind, target)
    y_tr, y_te = make_labels(kind, target, train, d_tr, d_te)
    X_tr, X_te, w = d_tr[num + cat], d_te[num + cat], recency_weights(d_tr)
    if kind == "bin" and pos_share(y_tr) is None:
        return []
    rows = []
    for name, (model, space) in (REG_MODELS if kind == "reg" else CLF_MODELS).items():
        model = clone(model)
        if name == "XGBoost" and kind == "bin":
            model.set_params(scale_pos_weight=(1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("model", model)])
        n_iter = 10 if name in ("Ridge", "Lasso", "Logistic") else N_ITER
        search = RandomizedSearchCV(pipe, {f"model__{k}": v for k, v in space.items()}, n_iter=n_iter, cv=INNER_CV,
                                    scoring="r2" if kind == "reg" else "average_precision", random_state=SEED,
                                    n_jobs=1, refit=False, error_score=np.nan)
        t0 = time.time()
        search.fit(X_tr, y_tr, model__sample_weight=w)
        best = search.best_index_
        best_params = {k.replace("model__", ""): v for k, v in search.cv_results_["params"][best].items()}
        tuned = clone(pipe).set_params(**{f"model__{k}": v for k, v in best_params.items()})
        fitted = clone(tuned).fit(X_tr, y_tr, model__sample_weight=w)
        row = {"setup": setup, "rule": rule, "scenario": scen, "task": task, "model": name,
               "n_train": len(y_tr), "n_test": len(y_te),
               "cv_mean": search.cv_results_["mean_test_score"][best], "cv_std": search.cv_results_["std_test_score"][best],
               "best_params": str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best_params.items()})}
        if kind == "reg":
            p_tr, p_te = fitted.predict(X_tr), fitted.predict(X_te)
            row.update(train_R2=r2_score(y_tr, p_tr), test_R2=r2_score(y_te, p_te), test_MAE=mean_absolute_error(y_te, p_te),
                       test_RMSE=float(np.sqrt(np.mean((y_te - p_te) ** 2))), test_rho=spearmanr(y_te, p_te).correlation)
        else:
            oof = oof_predictions(tuned, X_tr, y_tr, w, kind); v = ~np.isnan(oof)
            thr = f1_threshold(oof[v], y_tr[v])
            p_tr, p_te = fitted.predict_proba(X_tr)[:, 1], fitted.predict_proba(X_te)[:, 1]
            yhat = (p_te >= thr).astype(int)
            row.update(threshold=thr, train_ROC_AUC=roc_auc_score(y_tr, p_tr), test_ROC_AUC=roc_auc_score(y_te, p_te),
                       test_PR_AUC=average_precision_score(y_te, p_te), test_F1=f1_score(y_te, yhat, zero_division=0),
                       test_precision=precision_score(y_te, yhat, zero_division=0), test_recall=recall_score(y_te, yhat, zero_division=0),
                       test_base_rate=y_te.mean())
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
    return rows


def pos_share(y):
    return y.mean() if 0 < y.mean() < 1 else None


RESULTS = os.path.join(OUT_DIR, "exp2_all_models.csv")
done = pd.read_csv(RESULTS) if os.path.exists(RESULTS) else pd.DataFrame()
done_keys = set(map(tuple, done[["setup", "rule", "scenario", "task"]].values)) if len(done) else set()
t_all = time.time()
for setup in SETUPS:
    for rule in RULES:
        for scen in SCENARIO_LIST:
            for task in TASKS:
                if (setup, rule, scen, task) in done_keys:
                    continue
                t0 = time.time()
                rows = run_one(setup, rule, scen, task)
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(RESULTS, index=False)
                print(f"{setup} | {rule} | {scen} | {task}: {time.time() - t0:.0f}s", flush=True)
print(f"all combinations done ({(time.time() - t_all) / 60:.0f} min this run)")
res = pd.read_csv(RESULTS)'''))

cells.append(md("""## 9. Results

### 9.1 Best model per combination (chosen by cross-validation)"""))
cells.append(code('''res["metric_cv"] = res["cv_mean"]
best = res.loc[res.groupby(["setup", "rule", "scenario", "task"])["metric_cv"].idxmax()].reset_index(drop=True)
best.to_csv(os.path.join(OUT_DIR, "exp2_best_by_cv.csv"), index=False)
reg_cols = ["setup", "rule", "scenario", "task", "model", "cv_mean", "cv_std", "train_R2", "test_R2", "test_MAE", "test_rho"]
clf_cols = ["setup", "rule", "scenario", "model", "cv_mean", "cv_std", "train_ROC_AUC", "test_ROC_AUC", "test_PR_AUC",
            "test_base_rate", "test_F1", "test_recall", "test_precision"]
print("Regression (cv_mean = CV R²)")
display(best[best["task"] != "is_severe"][reg_cols].round(3))
print("is_severe (cv_mean = CV PR-AUC)")
display(best[best["task"] == "is_severe"][clf_cols].round(3))'''))

cells.append(md("""### 9.2 Which model wins most often?"""))
cells.append(code('''wins = best.groupby(["task", "model"]).size().unstack(fill_value=0)
display(wins)'''))

cells.append(md("""### 9.3 All models side by side, scenario A (pre-event)

Every tuned model, not only the selected one, so the comparison between model families is visible."""))
cells.append(code('''view = res[(res["rule"] == "R2_strict") & (res["scenario"] == "A_pre_event")]
for task in TASKS:
    v = view[view["task"] == task]
    cols = (["setup", "model", "cv_mean", "cv_std", "train_R2", "test_R2", "test_MAE", "test_rho"] if task != "is_severe"
            else ["setup", "model", "cv_mean", "cv_std", "test_ROC_AUC", "test_PR_AUC", "test_F1"])
    print(task); display(v[cols].round(3).reset_index(drop=True))'''))

cells.append(md("""### 9.4 Plot: selected model per scenario and setup

Test score of the CV-selected model for each scenario and setup (strict rule)."""))
cells.append(code('''fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, (task, col, title) in zip(axes, [("customers_reg", "test_R2", "Customers: test R²"), ("duration_reg", "test_R2", "Duration: test R²"),
                                          ("is_severe", "test_ROC_AUC", "Severe: test ROC-AUC")]):
    b = best[best["task"] == task]
    labels = ["A (pre-event)" if s[0] == "A" else "B (with cause)" for r in RULES for s in SCENARIO_LIST]
    x = np.arange(len(labels)); wdt = 0.38
    for k, (setup, colr) in enumerate([("2a_main", "#2a78d6"), ("2b_oe417_only", "#eb6834")]):
        vals = [b[(b["setup"] == setup) & (b["rule"] == r) & (b["scenario"] == s)][col].mean() for r in RULES for s in SCENARIO_LIST]
        ax.bar(x + (k - 0.5) * wdt, vals, wdt, color=colr, label={"2a_main": "2a main (test 2022-23)", "2b_oe417_only": "2b OE-417 only (test 2021-23)"}[setup])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(0, color="#8a8984", lw=0.8); ax.grid(axis="y", color="#e6e5e1")
axes[0].legend(fontsize=8)
fig.text(0.01, 0.0, "Strict storm rule (R2). Scenario A: pre-event information only; B: cause also known.", fontsize=8, color="#666")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp2_selected_models.png"), dpi=150); plt.show()'''))

cells.append(md("""### 9.5 Setup 2a vs 2b (selected models)"""))
cells.append(code('''key = ["rule", "scenario", "task"]
a = best[best["setup"] == "2a_main"].set_index(key); b = best[best["setup"] == "2b_oe417_only"].set_index(key)
cmp = pd.DataFrame({"2a test R² / ROC-AUC": np.where(a.index.get_level_values("task") == "is_severe", a["test_ROC_AUC"], a["test_R2"]),
                    "2b test R² / ROC-AUC": np.where(b.reindex(a.index).index.get_level_values("task") == "is_severe",
                                                     b.reindex(a.index)["test_ROC_AUC"], b.reindex(a.index)["test_R2"]),
                    "2a model": a["model"], "2b model": b.reindex(a.index)["model"]}, index=a.index)
cmp.to_csv(os.path.join(OUT_DIR, "exp2_setup_comparison.csv"))
display(cmp.round(3))'''))

cells.append(md("""## 10. Notes for interpretation

- The selected model is chosen by cross-validation on the training years, never by the test score.
- Test periods are short (2 or 3 years), so differences of a few hundredths are within the noise shown in Experiment 1.
- Scenario B uses the reported cause, which is known only after the event; it is an upper reference, not an operational setting.
- PR-AUC depends on the share of severe outages in the test period (`test_base_rate`); ROC-AUC does not."""))

write(cells, DST, meta())
