"""Train the deployed models and write the .pkl files used by the Flask app.

Runs the same data pipeline as the final-stage experiments (cells 1-41 of the
progress-stage notebook), then fits the models chosen by cross-validation in
Experiment 2 (strict storm rule, scenario A, train before 2022) and the
calibration chosen in Experiment 7 (isotonic + prior correction).

This script needs the raw data in po1/Data and is NOT part of the deployed app.
Run it from anywhere with the tf_cpu environment:

    /opt/anaconda3/envs/tf_cpu/bin/python app/training/train_model.py
"""
import ast
import os
import re
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
PO1 = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(APP_DIR, "models")
EXP2_BEST = os.path.join(PO1, "final_stage2", "exp2_model_search", "results", "exp2_best_by_cv.csv")
sys.path.insert(0, os.path.join(PO1, "final_stage2", "build"))
from common import data_cells  # noqa: E402  (same pipeline cells as every final-stage notebook)

# Top 5 features over all three targets (Experiment 5 importance, normalised per target and summed)
TOP5 = ["hist_state_storm_cust", "nws_watch_active", "noaa_n_zones", "nws_hours_since_warning", "noaa_storm_count"]
SPLIT, END = "2022-01-01", "2024-01-01"     # train before 2022 (as in the report); 2024 excluded
DEFAULT_YEARS = (2019, 2023)                # recent outages used to auto-fill the other features


def run_pipeline():
    """Execute the notebook's data cells and return their namespace (tables, feature_sets, ...)."""
    build_dir = os.path.join(HERE, "_build")
    os.makedirs(build_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(build_dir)                     # the pipeline finds po1/Data by walking up from here
    ns = {"display": lambda *a, **k: None, "__name__": "pipeline"}
    try:
        for c in data_cells():
            if c["cell_type"] == "code":
                exec("".join(c["source"]), ns)
    finally:
        os.chdir(cwd)
        shutil.rmtree(build_dir, ignore_errors=True)
    return ns


def main():
    t0 = time.time()
    ns = run_pipeline()
    print(f"pipeline done in {(time.time() - t0) / 60:.1f} min")
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline

    feature_sets, build_preprocessor = ns["feature_sets"], ns["build_preprocessor"]
    task_rows, make_labels, recency_weights, SEED = ns["task_rows"], ns["make_labels"], ns["recency_weights"], ns["SEED"]
    df = ns["tables"]["strict"]
    df = df[df["outage_start"] < pd.Timestamp(END)]
    train = df[df["outage_start"] < pd.Timestamp(SPLIT)]
    test = df[df["outage_start"] >= pd.Timestamp(SPLIT)]
    NUM, CAT = feature_sets(df, "A_pre_event")
    assert all(f in NUM for f in TOP5), "top-5 features missing from the feature set"
    print(f"features: {len(NUM)} numeric + {len(CAT)} categorical; train {len(train)}, test {len(test)} outages")

    best = pd.read_csv(EXP2_BEST)
    best = best[(best["setup"] == "2a_main") & (best["rule"] == "R2_strict") & (best["scenario"] == "A_pre_event")].set_index("task")
    params = lambda task: ast.literal_eval(re.sub("np[.][a-z0-9]+[(]([^)]*)[)]", lambda m: m.group(1), best.loc[task, "best_params"]))
    base = {"RandomForest": RandomForestRegressor(random_state=SEED, n_jobs=-1),
            "Logistic": LogisticRegression(class_weight="balanced", max_iter=5000)}

    def fit(task, d, y):
        m = clone(base[best.loc[task, "model"]]).set_params(**params(task))
        return Pipeline([("pre", build_preprocessor(NUM, CAT)), ("model", m)]).fit(d[NUM + CAT], y, model__sample_weight=recency_weights(d))

    models, metrics = {}, {}
    # --- regression: customers affected and duration
    for task, target in [("customers_reg", "log_customers"), ("duration_reg", "log_duration")]:
        d_tr = task_rows(train, "reg", target).sort_values("outage_start", kind="stable")
        d_te = task_rows(test, "reg", target)
        y_tr, y_te = make_labels("reg", target, train, d_tr, d_te)
        models[task] = fit(task, d_tr, y_tr)
        metrics[task] = {"model": best.loc[task, "model"], "cv_r2": float(best.loc[task, "cv_mean"]),
                         "test_r2": float(r2_score(y_te, models[task].predict(d_te[NUM + CAT]))),
                         "n_train": len(d_tr), "n_test": len(d_te)}
        print(task, metrics[task])

    # --- severe: model + isotonic calibration with prior correction (Experiment 7)
    d_tr = task_rows(train, "bin", None).sort_values("outage_start", kind="stable")
    d_te = task_rows(test, "bin", None)
    y_tr, y_te = [np.asarray(v) for v in make_labels("bin", None, train, d_tr, d_te)]
    oof, fold = np.full(len(y_tr), np.nan), np.full(len(y_tr), -1)
    for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=4).split(d_tr)):
        oof[b] = fit("is_severe", d_tr.iloc[a], y_tr[a]).predict_proba(d_tr.iloc[b][NUM + CAT])[:, 1]
        fold[b] = k
    v = fold >= 0
    P, Y, FOLD = oof[v], y_tr[v], fold[v]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(P, Y)
    pi_fit, pi_recent = float(Y.mean()), float(Y[FOLD == 3].mean())
    models["is_severe"] = fit("is_severe", d_tr, y_tr)
    p_te = models["is_severe"].predict_proba(d_te[NUM + CAT])[:, 1]

    def calibrate(p):
        p = np.clip(iso.predict(p), 1e-4, 1 - 1e-4)
        a, b = p * pi_recent / pi_fit, (1 - p) * (1 - pi_recent) / (1 - pi_fit)
        return a / (a + b)

    metrics["is_severe"] = {"model": best.loc["is_severe", "model"], "cv_pr_auc": float(best.loc["is_severe", "cv_mean"]),
                            "test_roc_auc": float(roc_auc_score(y_te, p_te)), "test_mean_calibrated": float(calibrate(p_te).mean()),
                            "test_severe_share": float(y_te.mean()), "n_train": len(d_tr), "n_test": len(d_te)}
    print("is_severe", metrics["is_severe"])

    # --- defaults for auto-filling the features the user does not enter
    recent = df[df["outage_start"].dt.year.between(*DEFAULT_YEARS)]

    def summarise(rows):
        num = rows[NUM].median(numeric_only=True).reindex(NUM)
        cat = {c: (rows[c].dropna().mode().iloc[0] if rows[c].notna().any() else None) for c in CAT}
        return {"num": {k: (None if pd.isna(x) else float(x)) for k, x in num.items()}, "cat": cat, "n": int(len(rows))}

    defaults, population = {}, {}
    for state in sorted(df["clean_state"].dropna().unique()):
        s_all, s_recent = df[df["clean_state"] == state], recent[recent["clean_state"] == state]
        pop = s_all.sort_values("outage_start")["population"].dropna()
        if pop.empty:                       # customers per 100k cannot be turned into a count
            print(f"skipped {state}: no population value")
            continue
        population[state] = float(np.expm1(pop.iloc[-1]))  # the pipeline stores log(population)
        defaults[state] = {}
        for storm in (0, 1):
            pick = lambda d: d[(d["noaa_storm_count"] > 0) == bool(storm)]
            rows = next((r for r in (pick(s_recent), pick(s_all), pick(recent)) if len(r) >= 3), pick(df))
            defaults[state][storm] = summarise(rows)
            defaults[state][storm]["cat"]["clean_state"] = state

    ranges = {f: {"p99": float(df[f].quantile(0.99)), "max": float(df[f].max())} for f in TOP5}
    meta = {"num": NUM, "cat": CAT, "top5": TOP5, "defaults": defaults, "population": population, "ranges": ranges,
            "calibration": {"iso": iso, "pi_fit": pi_fit, "pi_recent": pi_recent}, "metrics": metrics,
            "train_period": f"2000 to {SPLIT[:4]} (exclusive)", "default_years": DEFAULT_YEARS,
            "versions": {"sklearn": __import__("sklearn").__version__, "pandas": pd.__version__, "numpy": np.__version__}}

    os.makedirs(MODEL_DIR, exist_ok=True)
    for task, name in [("customers_reg", "customers_model.pkl"), ("duration_reg", "duration_model.pkl"), ("is_severe", "severe_model.pkl")]:
        joblib.dump(models[task], os.path.join(MODEL_DIR, name), compress=3)
    joblib.dump(meta, os.path.join(MODEL_DIR, "app_metadata.pkl"), compress=3)
    for f in sorted(os.listdir(MODEL_DIR)):
        print(f"{f}: {os.path.getsize(os.path.join(MODEL_DIR, f)) / 1e6:.1f} MB")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
