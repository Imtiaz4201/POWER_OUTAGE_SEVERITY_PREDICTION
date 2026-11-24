"""Build final_stage2/exp4_per_target_selection/exp4_per_target_selection.ipynb."""
import sys
sys.path.insert(0, "/Users/imtiaz/Documents/po1/final_stage2/build")
from common import md, code, data_cells, write, meta
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp4_per_target_selection/exp4_per_target_selection.ipynb"

INTRO = """# Final stage, Experiment 4: per-target choice of training years and history features

**Problem statement:** *Can we predict how severe a major power outage will become, using historical outage data and recent storm information available when the outage starts?*

**Why this experiment:** Experiment 3 suggested that severe detection improves when the oldest years (2000–2005, no NWS data) and the outage-history features are left out, while the regression targets did not change. Each target has its own independent model, so the same **procedure** is applied to all three targets and the data decides per target.

**Design (decided before running):**

| | |
|---|---|
| Setup | strict storm rule (R2), scenario A (pre-event), main split: train before 2022, test 2022–2023 (2024 excluded) |
| Models | the cross-validation-selected tuned model of Experiment 2 for each target (same settings) |
| Options per target | 1: all training years, with outage-history features · 2: all years, without · 3: 2006 onward, with · 4: 2006 onward, without |
| Selection | time-based cross-validation on the training years (4 folds). **The validation folds are identical for all options**; only the training rows (all years vs 2006 onward) and the features change. The option with the best cross-validation score is chosen. |
| Test | the chosen option is refit and scored **once** on 2022–2023 |
| Labels | is_severe thresholds (Q1 / Q3) from all training years for every option, so the definition of "severe" does not change |

"Outage-history features" = the state, global and storm-status history group of Experiment 3 (`HIST_NUM`, `RECENT_HIST`, `STORM_TE`)."""

cells = [md(INTRO)] + data_cells()

cells.append(md("""## 7. Models from Experiment 2 and the four options"""))
cells.append(code('''import ast
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score
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
    text = re.sub("np[.][a-z0-9]+[(]([^)]*)[)]", lambda m: m.group(1), sel.loc[task, "best_params"])  # np.float64(0.07) -> 0.07
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
NUM_ALL, CAT = feature_sets(df, SCEN)
HISTORY = set(HIST_NUM + RECENT_HIST + STORM_TE)
OPTIONS = {
    "1: all years, with history": (2000, True),
    "2: all years, without history": (2000, False),
    "3: 2006 onward, with history": (2006, True),
    "4: 2006 onward, without history": (2006, False),
}
CV = TimeSeriesSplit(n_splits=4)
print("history features removed in options 2 and 4:", sorted(c for c in NUM_ALL if c in HISTORY))'''))

cells.append(md("""## 8. Cross-validation of the four options (training years only)"""))
cells.append(code('''def task_data(task):
    kind, target = TASKS[task]
    d_tr = task_rows(TRAIN, kind, target).sort_values("outage_start", kind="stable")
    d_te = task_rows(TEST, kind, target)
    y_tr, y_te = make_labels(kind, target, TRAIN, d_tr, d_te)
    return kind, d_tr, d_te, np.asarray(y_tr), np.asarray(y_te)


def metric(kind, y, p):
    return r2_score(y, p) if kind == "reg" else average_precision_score(y, p)


def fit(task, kind, d, y, num):
    pipe = Pipeline([("pre", build_preprocessor(num, CAT)), ("model", tuned_model(task, y))])
    return pipe.fit(d[num + CAT], y, model__sample_weight=recency_weights(d))


cv_rows, oof_store = [], {}
t0 = time.time()
for task in TASKS:
    kind, d_tr, d_te, y_tr, y_te = task_data(task)
    years = d_tr["outage_start"].dt.year.values
    for opt, (y0, with_hist) in OPTIONS.items():
        num = [c for c in NUM_ALL if with_hist or c not in HISTORY]
        scores, oof = [], np.full(len(y_tr), np.nan)
        for a, b in CV.split(d_tr):                      # same validation folds b for every option
            a = a[years[a] >= y0]                         # only the training part changes
            m = fit(task, kind, d_tr.iloc[a], y_tr[a], num)
            p = predict(m, kind, d_tr.iloc[b][num + CAT])
            oof[b] = p
            scores.append(metric(kind, y_tr[b], p))
        oof_store[(task, opt)] = oof
        cv_rows.append({"task": task, "option": opt, "n_features": len(num) + len(CAT),
                        "cv_mean": np.mean(scores), "cv_std": np.std(scores), "folds": np.round(scores, 3).tolist()})
    print(f"{task} done ({time.time() - t0:.0f}s)", flush=True)
cvt = pd.DataFrame(cv_rows)
cvt["chosen"] = cvt.groupby("task")["cv_mean"].transform("max") == cvt["cv_mean"]
cvt.to_csv(os.path.join(OUT_DIR, "exp4_cv_options.csv"), index=False)
print("cv_mean = R² (customers, duration) or PR-AUC (is_severe), the same metrics used to select models in Experiment 2")
display(cvt.round(3))'''))

cells.append(md("""## 9. Test the chosen option once (2022–2023)

The chosen option is refit on its training rows and scored once on the test period. The default option (1) is shown next to it for comparison; it is the same model and data as in Experiment 2."""))
cells.append(code('''def f1_threshold(p, y):
    grid = np.linspace(0.05, 0.95, 91)
    return grid[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid]))]


test_rows = []
for task in TASKS:
    kind, d_tr, d_te, y_tr, y_te = task_data(task)
    years = d_tr["outage_start"].dt.year.values
    chosen = cvt[(cvt["task"] == task) & cvt["chosen"]]["option"].iloc[0]
    for opt in dict.fromkeys(["1: all years, with history", chosen]):
        y0, with_hist = OPTIONS[opt]
        num = [c for c in NUM_ALL if with_hist or c not in HISTORY]
        keep = years >= y0
        m = fit(task, kind, d_tr[keep], y_tr[keep], num)
        p = predict(m, kind, d_te[num + CAT])
        row = {"task": task, "option": opt, "role": "chosen by CV" if opt == chosen else "default (Experiment 2)",
               "n_train": int(keep.sum()), "n_test": len(y_te)}
        if kind == "reg":
            row.update(test_R2=r2_score(y_te, p), test_MAE=mean_absolute_error(y_te, p), test_rho=spearmanr(y_te, p).correlation)
        else:
            oof = oof_store[(task, opt)]; v = ~np.isnan(oof)
            thr = f1_threshold(oof[v], y_tr[v]); yhat = (p >= thr).astype(int)
            row.update(test_ROC_AUC=roc_auc_score(y_te, p), test_PR_AUC=average_precision_score(y_te, p), threshold=thr,
                       test_F1=f1_score(y_te, yhat), test_recall=recall_score(y_te, yhat), test_precision=precision_score(y_te, yhat),
                       test_base_rate=y_te.mean())
        test_rows.append(row)
tst = pd.DataFrame(test_rows)
tst.to_csv(os.path.join(OUT_DIR, "exp4_test.csv"), index=False)
display(tst.round(3))'''))

cells.append(md("""## 10. Plot: cross-validation score of each option"""))
cells.append(code('''fig, axes = plt.subplots(1, 3, figsize=(16, 4))
names = {"customers_reg": "Customers: CV R²", "duration_reg": "Duration: CV R²", "is_severe": "Severe: CV PR-AUC"}
for ax, task in zip(axes, TASKS):
    s = cvt[cvt["task"] == task]
    colors = ["#2a78d6" if c else "#b9cfee" for c in s["chosen"]]
    ax.bar(range(len(s)), s["cv_mean"], yerr=s["cv_std"], color=colors, capsize=3, ecolor="#8a8984")
    for i, v in enumerate(s["cv_mean"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(s))); ax.set_xticklabels([o.split(":")[0] for o in s["option"]])
    ax.set_title(names[task], loc="left", fontweight="bold"); ax.grid(axis="y", color="#e6e5e1")
fig.text(0.01, 0.0, "Options: 1 all years + history · 2 all years, no history · 3 2006+ with history · 4 2006+ no history. Dark bar = chosen by CV; error bar = std over folds.",
         fontsize=8, color="#666")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp4_cv_options.png"), dpi=150); plt.show()'''))

cells.append(md("""## 11. Notes for interpretation

- The choice is made by cross-validation on the training years only; the test period is used once, for the chosen option (and the default for comparison).
- The same procedure is applied to every target; the chosen option may differ per target because each target is a separate model.
- Differences smaller than the fold spread (`cv_std`) are within noise; in that case the simpler or default option is equally good."""))

write(cells, DST, meta())
