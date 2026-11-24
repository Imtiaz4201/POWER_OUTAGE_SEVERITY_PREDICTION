"""Build final_stage2/exp7_rq2_reliability/exp7_rq2_reliability.ipynb."""
import sys
sys.path.insert(0, "/Users/imtiaz/Documents/po1/final_stage2/build")
from common import md, code, data_cells, write, meta
DST = "/Users/imtiaz/Documents/po1/final_stage2/exp7_rq2_reliability/exp7_rq2_reliability.ipynb"

INTRO = """# Final stage, Experiment 7 (RQ2): reliable, cost-sensitive severe-outage detection

**RQ2:** *Can calibrated, cost-sensitive models detect severe outages more reliably, especially where false negatives carry greater operational consequences?*

**Background:** in the progress stage the severe-detection model beat simple baselines, but a threshold tuned for a 3:1 cost on the training years did not transfer well to the test years, because severe outages became rarer (about 50% of training outages, about 30% in the test years).

**Design (decided before running):**

| | |
|---|---|
| Setup | strict storm rule (R2), scenario A (pre-event), main split: train before 2022, test 2022–2023 (2024 excluded) |
| Model | the cross-validation-selected tuned model of Experiment 2 for is_severe |
| Data for calibration and thresholds | time-based cross-validation predictions on the training years (4 folds). "Recent years" = the latest fold (about late 2019–2021). The test years are never used to fit or choose anything. |
| 1. Calibration methods | raw · Platt · isotonic · Platt + prior correction · isotonic + prior correction · Platt fitted on recent years only |
| Prior correction | adjusts the probabilities from the severe share of the calibration data to the severe share of the **recent training years** (known at deployment; the test share is not used) |
| Choosing the method | fit on the earlier folds, score on the latest fold (Brier score); the best method is refitted on all folds and applied to the test years once |
| 2. Threshold strategies | (a) F1 threshold (progress stage) · (b) cost-tuned on all training years (progress stage) · (c) cost-tuned on recent years · (d) theoretical threshold 1 / (1 + c) on the chosen calibrated probabilities, where c = cost of a missed severe outage in false alarms |
| 3. Cost curve | expected cost per outage on 2022–2023 for c = 1 … 10, with baselines (always / never severe) and the best possible threshold on the test years (reference only, not usable in practice) |
| Uncertainty | paired bootstrap (2,000 resamples) on the test outages |

Expected cost per outage = (c × missed severe outages + false alarms) / number of outages."""

cells = [md(INTRO)] + data_cells()

cells.append(md("""## 7. Model and predictions"""))
cells.append(code('''import ast
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

EXP2 = os.path.join(os.path.dirname(os.getcwd()), "exp2_model_search", "results", "exp2_best_by_cv.csv")
sel = pd.read_csv(EXP2)
sel = sel[(sel["setup"] == "2a_main") & (sel["rule"] == "R2_strict") & (sel["scenario"] == "A_pre_event") & (sel["task"] == "is_severe")].iloc[0]
BASE = {"Logistic": LogisticRegression(class_weight="balanced", max_iter=5000),
        "RandomForest": RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0, eval_metric="logloss")}
params = ast.literal_eval(re.sub("np[.][a-z0-9]+[(]([^)]*)[)]", lambda m: m.group(1), sel["best_params"]))
print("model:", sel["model"], params)

RULE, SCEN = "strict", "A_pre_event"
SPLIT, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
df = tables[RULE]
TRAIN, TEST = df[df["outage_start"] < SPLIT], df[(df["outage_start"] >= SPLIT) & (df["outage_start"] < END)]
NUM, CAT = feature_sets(df, SCEN)
d_tr = task_rows(TRAIN, "bin", None).sort_values("outage_start", kind="stable")
d_te = task_rows(TEST, "bin", None)
y_tr, y_te = [np.asarray(v) for v in make_labels("bin", None, TRAIN, d_tr, d_te)]


def fit_model(d, y):
    m = clone(BASE[sel["model"]]).set_params(**params)
    if sel["model"] == "XGBoost":
        m.set_params(scale_pos_weight=(1 - y.mean()) / max(y.mean(), 1e-6))
    return Pipeline([("pre", build_preprocessor(NUM, CAT)), ("model", m)]).fit(d[NUM + CAT], y, model__sample_weight=recency_weights(d))


oof, fold = np.full(len(y_tr), np.nan), np.full(len(y_tr), -1)
for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=4).split(d_tr)):
    oof[b] = predict(fit_model(d_tr.iloc[a], y_tr[a]), "bin", d_tr.iloc[b][NUM + CAT]); fold[b] = k
p_te = predict(fit_model(d_tr, y_tr), "bin", d_te[NUM + CAT])
v = fold >= 0
P, Y, FOLD = oof[v], y_tr[v], fold[v]          # cross-validation predictions on the training years
years = d_tr["outage_start"].dt.year.values[v]

share = pd.DataFrame({"outages": [len(y_tr), int((FOLD == 3).sum()), len(y_te)],
                      "severe share": [y_tr.mean(), Y[FOLD == 3].mean(), y_te.mean()]},
                     index=["all training years (2000-2021)", f"recent training years (latest fold, {years[FOLD == 3].min()}-{years[FOLD == 3].max()})",
                            "test years (2022-2023)"])
share.to_csv(os.path.join(OUT_DIR, "exp7_severe_share.csv"))
display(share.round(3))
print(f"ranking quality (unchanged by calibration): CV ROC-AUC {roc_auc_score(Y, P):.3f}, test ROC-AUC {roc_auc_score(y_te, p_te):.3f}, "
      f"test PR-AUC {average_precision_score(y_te, p_te):.3f}")'''))

cells.append(md("""## 8. Calibration

A calibrated probability means: of the outages given "40% severe", about 40% really are severe.

- **Brier score:** mean squared difference between probability and outcome (lower is better).
- **ECE (expected calibration error):** average gap between predicted and observed severe share over probability bins (lower is better).
- **Mean predicted vs observed share:** shows whether the probabilities are too high or too low overall."""))
cells.append(code('''def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def prior_shift(p, pi_old, pi_new):
    """Re-weight probabilities from severe share pi_old to pi_new."""
    a, b = p * pi_new / pi_old, (1 - p) * (1 - pi_new) / (1 - pi_old)
    return a / (a + b)


def make_calibrator(method, p_fit, y_fit, recent):
    """recent = boolean mask of the most recent rows of the fitting data."""
    pi_fit, pi_recent = y_fit.mean(), y_fit[recent].mean()
    if method == "raw":
        return lambda p: p
    if method.startswith("Platt, recent"):
        lr = LogisticRegression(C=1e6).fit(logit(p_fit[recent])[:, None], y_fit[recent])
        return lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]
    if method.startswith("Platt"):
        lr = LogisticRegression(C=1e6).fit(logit(p_fit)[:, None], y_fit)
        base = lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]
    else:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_fit, y_fit)
        base = lambda p: iso.predict(p)
    if method.endswith("prior correction"):
        return lambda p: prior_shift(np.clip(base(p), 1e-4, 1 - 1e-4), pi_fit, pi_recent)
    return base


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def cal_scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"Brier": brier_score_loss(y, p), "ECE": ece(y, p), "log loss": log_loss(y, p),
            "mean predicted": p.mean(), "observed share": y.mean()}


METHODS = ["raw", "Platt", "isotonic", "Platt + prior correction", "isotonic + prior correction", "Platt, recent years only"]

# Stage 1 (choose the method without the test years): fit on folds 0-2, score on fold 3 (most recent training years)
fit_m, val_m = FOLD <= 2, FOLD == 3
rows = []
for m in METHODS:
    cal = make_calibrator(m, P[fit_m], Y[fit_m], FOLD[fit_m] == 2)
    rows.append({"method": m, **cal_scores(Y[val_m], cal(P[val_m]))})
val_tab = pd.DataFrame(rows).set_index("method")
CHOSEN = val_tab["Brier"].idxmin()
val_tab.to_csv(os.path.join(OUT_DIR, "exp7_calibration_validation.csv"))
print(f"Validation on the most recent training years ({years[val_m].min()}-{years[val_m].max()}, n = {val_m.sum()}); chosen method: {CHOSEN}")
display(val_tab.round(3))

# Stage 2: refit every method on all folds (recent = fold 3) and apply to the test years once
CAL = {m: make_calibrator(m, P, Y, FOLD == 3) for m in METHODS}
test_tab = pd.DataFrame([{"method": m, "chosen": m == CHOSEN, **cal_scores(y_te, CAL[m](p_te))} for m in METHODS]).set_index("method")
test_tab.to_csv(os.path.join(OUT_DIR, "exp7_calibration_test.csv"))
print(f"Test years 2022-2023 (n = {len(y_te)})")
display(test_tab.round(3))'''))

cells.append(md("""### 8.1 Reliability plot (test years)

Outages are sorted into five equal groups by predicted probability; each point shows the mean predicted probability and the observed share of severe outages. Points on the dashed line are perfectly calibrated."""))
cells.append(code('''fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", lw=1, label="perfect")
shown = list(dict.fromkeys(["raw", "Platt", CHOSEN]))
rel_rows = []
for m, col, mk in zip(shown, ["#8a8984", "#2a78d6", "#eb6834"], ["o", "s", "D"]):
    p = CAL[m](p_te)
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False).values
    xs, ys = [p[q == b].mean() for b in range(5)], [y_te[q == b].mean() for b in range(5)]
    ax.plot(xs, ys, marker=mk, color=col, label=m + (" (chosen)" if m == CHOSEN else ""))
    rel_rows += [{"method": m, "group": b + 1, "mean predicted": xs[b], "observed share": ys[b]} for b in range(5)]
pd.DataFrame(rel_rows).to_csv(os.path.join(OUT_DIR, "exp7_reliability.csv"), index=False)
ax.set(xlabel="predicted probability of severe", ylabel="observed share severe", xlim=(0, 1), ylim=(0, 1))
ax.set_title("Reliability on 2022-2023", loc="left", fontweight="bold"); ax.legend(fontsize=8); ax.grid(color="#e6e5e1")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp7_reliability.png"), dpi=150); plt.show()'''))

cells.append(md("""## 9. Decision thresholds

Four strategies, all fixed before looking at the test years:

| | Strategy | Probabilities | Threshold |
|---|---|---|---|
| a | F1 threshold (progress stage) | raw | best F1 on all cross-validation predictions; same for every cost |
| b | Cost-tuned, all training years (progress stage) | Platt | lowest cost on all cross-validation predictions |
| c | Cost-tuned, recent years | chosen calibration | lowest cost on the most recent fold |
| d | Theoretical threshold | chosen calibration | 1 / (1 + c): flag an outage when its probability of being severe exceeds this value |

Baselines: always "not severe", always "severe". Reference: the best possible threshold on the test years (not usable in practice)."""))
cells.append(code('''def cost(y, flag, c):
    return (c * ((y == 1) & ~flag).sum() + ((y == 0) & flag).sum()) / len(y)


GRID = np.linspace(0.01, 0.99, 99)
def cost_threshold(p, y, c):
    return GRID[int(np.argmin([cost(y, p >= t, c) for t in GRID]))]


recent = FOLD == 3
p_raw_cv, p_platt_cv, p_ch_cv = P, CAL["Platt"](P), CAL[CHOSEN](P)
p_raw_te, p_platt_te, p_ch_te = p_te, CAL["Platt"](p_te), CAL[CHOSEN](p_te)
T_F1 = GRID[int(np.argmax([f1_score(Y, p_raw_cv >= t, zero_division=0) for t in GRID]))]


def strategies(c):
    """name -> (threshold, test probabilities)"""
    return {
        "a. F1 threshold (progress stage)": (T_F1, p_raw_te),
        "b. cost-tuned, all training years (progress stage)": (cost_threshold(p_platt_cv, Y, c), p_platt_te),
        "c. cost-tuned, recent years": (cost_threshold(p_ch_cv[recent], Y[recent], c), p_ch_te),
        "d. theoretical threshold 1/(1+c)": (1 / (1 + c), p_ch_te),
    }


rows = []
for c in COST_RATIOS:
    rows.append({"cost ratio": f"{c}:1", "strategy": "baseline: always not severe", "threshold": np.nan, "cost": cost(y_te, np.zeros(len(y_te), bool), c),
                 "recall": 0.0, "false-alarm rate": 0.0})
    rows.append({"cost ratio": f"{c}:1", "strategy": "baseline: always severe", "threshold": np.nan, "cost": cost(y_te, np.ones(len(y_te), bool), c),
                 "recall": 1.0, "false-alarm rate": 1.0})
    for name, (t, p) in strategies(c).items():
        flag = p >= t
        rows.append({"cost ratio": f"{c}:1", "strategy": name, "threshold": t, "cost": cost(y_te, flag, c),
                     "recall": flag[y_te == 1].mean(), "false-alarm rate": flag[y_te == 0].mean()})
    best_t = cost_threshold(p_ch_te, y_te, c)
    rows.append({"cost ratio": f"{c}:1", "strategy": "reference: best possible threshold on test (not usable)", "threshold": best_t,
                 "cost": cost(y_te, p_ch_te >= best_t, c), "recall": (p_ch_te >= best_t)[y_te == 1].mean(),
                 "false-alarm rate": (p_ch_te >= best_t)[y_te == 0].mean()})
thr = pd.DataFrame(rows)
thr.to_csv(os.path.join(OUT_DIR, "exp7_thresholds_test.csv"), index=False)
print(f"chosen calibration: {CHOSEN}; test years 2022-2023, n = {len(y_te)}, severe share {y_te.mean():.3f}")
for c in COST_RATIOS:
    print(f"cost ratio {c}:1"); display(thr[thr["cost ratio"] == f"{c}:1"].drop(columns="cost ratio").round(3).reset_index(drop=True))'''))

cells.append(md("""## 10. Cost curve: expected cost for cost ratios 1 to 10 (test years)"""))
cells.append(code('''CS = np.arange(1, 10.01, 0.5)
curve = {"always not severe": [], "always severe": [], "a. F1 threshold (progress stage)": [],
         "b. cost-tuned, all training years (progress stage)": [], "c. cost-tuned, recent years": [],
         "d. theoretical threshold 1/(1+c)": [], "best possible on test (reference)": []}
for c in CS:
    curve["always not severe"].append(cost(y_te, np.zeros(len(y_te), bool), c))
    curve["always severe"].append(cost(y_te, np.ones(len(y_te), bool), c))
    for name, (t, p) in strategies(c).items():
        curve[name].append(cost(y_te, p >= t, c))
    curve["best possible on test (reference)"].append(min(cost(y_te, p_ch_te >= t, c) for t in GRID))
cc = pd.DataFrame(curve, index=pd.Index(CS, name="cost ratio c"))
cc.to_csv(os.path.join(OUT_DIR, "exp7_cost_curve.csv"))
style = {"always not severe": ("#8a8984", ":"), "always severe": ("#8a8984", "--"), "a. F1 threshold (progress stage)": ("#eda100", "-"),
         "b. cost-tuned, all training years (progress stage)": ("#e87ba4", "-"), "c. cost-tuned, recent years": ("#1baf7a", "-"),
         "d. theoretical threshold 1/(1+c)": ("#2a78d6", "-"), "best possible on test (reference)": ("#0b0b0b", "-.")}
fig, ax = plt.subplots(figsize=(9.5, 5))
for name, (col, ls) in style.items():
    ax.plot(cc.index, cc[name], color=col, ls=ls, lw=2.4 if name.startswith("d.") else 1.5, label=name)
ax.set(xlabel="cost of a missed severe outage, in false alarms (c)", ylabel="expected cost per outage (lower is better)")
ax.set_title("Cost curve on 2022-2023", loc="left", fontweight="bold"); ax.legend(fontsize=8); ax.grid(color="#e6e5e1")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "exp7_cost_curve.png"), dpi=150); plt.show()
summary = pd.DataFrame({"mean cost over c = 1..10": cc.mean(), "times lowest among usable strategies": [
    int((cc.drop(columns="best possible on test (reference)").idxmin(axis=1) == k).sum()) for k in cc.columns]}).round(3)
summary.to_csv(os.path.join(OUT_DIR, "exp7_cost_curve_summary.csv"))
display(summary)'''))

cells.append(md("""## 11. Uncertainty: paired bootstrap on the test outages

The test period has few outages, so each difference is given with a 95% interval (2,000 resamples of the test outages). An interval that excludes 0 means the difference is unlikely to be chance."""))
cells.append(code('''rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(y_te), (2000, len(y_te)))


def boot(fn):
    vals = np.array([fn(i) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


rows = []
d0 = brier_score_loss(y_te, p_raw_te) - brier_score_loss(y_te, p_ch_te)
lo, hi = boot(lambda i: brier_score_loss(y_te[i], p_raw_te[i]) - brier_score_loss(y_te[i], p_ch_te[i]))
rows.append({"comparison": f"Brier: raw minus {CHOSEN}", "difference": d0, "ci_low": lo, "ci_high": hi})
for c in COST_RATIOS:
    s = strategies(c)
    fd = s["d. theoretical threshold 1/(1+c)"][1] >= s["d. theoretical threshold 1/(1+c)"][0]
    for other in ["a. F1 threshold (progress stage)", "b. cost-tuned, all training years (progress stage)"]:
        fo = s[other][1] >= s[other][0]
        lo, hi = boot(lambda i: cost(y_te[i], fo[i], c) - cost(y_te[i], fd[i], c))
        rows.append({"comparison": f"cost {c}:1: {other[:2]} minus d.", "difference": cost(y_te, fo, c) - cost(y_te, fd, c), "ci_low": lo, "ci_high": hi})
    base = min(cost(y_te, np.zeros(len(y_te), bool), c), cost(y_te, np.ones(len(y_te), bool), c))
    lo, hi = boot(lambda i: min(cost(y_te[i], np.zeros(len(i), bool), c), cost(y_te[i], np.ones(len(i), bool), c)) - cost(y_te[i], fd[i], c))
    rows.append({"comparison": f"cost {c}:1: best baseline minus d.", "difference": base - cost(y_te, fd, c), "ci_low": lo, "ci_high": hi})
bt = pd.DataFrame(rows)
bt["significant (interval excludes 0)"] = (bt["ci_low"] > 0) | (bt["ci_high"] < 0)
bt.to_csv(os.path.join(OUT_DIR, "exp7_bootstrap.csv"), index=False)
print("positive difference = strategy d (or the chosen calibration) is better")
display(bt.round(3))'''))

cells.append(md("""## 12. Notes for interpretation

- Calibration changes the probabilities, not the ranking, so ROC-AUC and PR-AUC stay (almost) the same; what changes is how well a probability such as 0.4 matches reality, and therefore which threshold is right.
- The prior correction uses the severe share of the recent **training** years. If the share keeps changing after training, some miscalibration remains.
- Strategy d needs no threshold tuning: with calibrated probabilities, the cost-optimal rule is to flag an outage when its probability exceeds 1 / (1 + c).
- "Best possible on test" uses the test outcomes and only shows how much room is left; it is not a usable strategy.
- The test period has 163 outages (49 severe); read differences together with the bootstrap intervals."""))

write(cells, DST, meta())
