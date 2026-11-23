"""Load the trained models and turn a few user inputs into the full feature row."""
import math
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# The five inputs shown in the form (the top-5 features over all three targets, Experiment 5).
INPUTS = {
    "hist_state_storm_cust": {
        "label": "Typical customers affected in past outages in this state",
        "unit": "per 100,000 residents",
        "help": "Usual size of earlier outages in the state with the same storm status. Pre-filled from the outage history.",
        "step": "1",
    },
    "nws_watch_active": {
        "label": "NWS watches in effect in the state",
        "unit": "count",
        "help": "Weather watches (e.g. tornado, flood, winter storm) active at the outage start.",
        "step": "1",
    },
    "noaa_n_zones": {
        "label": "Zones with an active storm (last 24 h)",
        "unit": "count",
        "help": "Number of forecast zones / counties in the state with a storm in the last 24 hours.",
        "step": "1",
    },
    "nws_hours_since_warning": {
        "label": "Hours since the last NWS warning",
        "unit": "hours (max 168)",
        "help": "Time since the most recent weather warning in the state; 168 = none in the last week.",
        "step": "0.5",
    },
    "noaa_storm_count": {
        "label": "Storm reports in the state (last 24 h)",
        "unit": "count",
        "help": "Storm events that began at least 1 h before the outage and were active in the last 24 hours. 0 = no storm.",
        "step": "1",
    },
}


class InputError(ValueError):
    """Raised for invalid user input; the message is shown to the user."""


class OutagePredictor:
    def __init__(self, model_dir):
        load = lambda name: joblib.load(os.path.join(model_dir, name))
        self.customers_model = load("customers_model.pkl")
        self.duration_model = load("duration_model.pkl")
        self.severe_model = load("severe_model.pkl")
        meta = load("app_metadata.pkl")
        self.num, self.cat = meta["num"], meta["cat"]
        self.defaults, self.population = meta["defaults"], meta["population"]
        self.ranges, self.metrics = meta["ranges"], meta["metrics"]
        self.default_years = meta["default_years"]
        cal = meta["calibration"]
        self._iso, self._pi_fit, self._pi_recent = cal["iso"], cal["pi_fit"], cal["pi_recent"]
        self.n_features = len(self.num) + len(self.cat)

    # ---------- helpers for the form
    def states(self):
        return [(s, s.title()) for s in sorted(self.defaults)]

    def prefill(self):
        """Typical values of the five inputs per state and storm status, in the units shown in the form."""
        out = {}
        for state, by_storm in self.defaults.items():
            out[state] = {}
            for storm, d in by_storm.items():
                vals = {f: d["num"].get(f) for f in INPUTS}
                if vals["hist_state_storm_cust"] is not None:
                    vals["hist_state_storm_cust"] = math.expm1(vals["hist_state_storm_cust"])
                out[state][int(storm)] = {k: (None if v is None else round(v, 1)) for k, v in vals.items()}
        return out

    # ---------- prediction
    def parse(self, form):
        state = (form.get("state") or "").strip().upper()
        if state not in self.defaults:
            raise InputError("Please choose a state.")
        try:
            start = datetime.fromisoformat((form.get("start") or "").strip())
        except ValueError:
            raise InputError("Please enter the outage start date and time.")
        values = {}
        for f, spec in INPUTS.items():
            raw = str(form.get(f, "")).strip()
            try:
                x = float(raw)
            except ValueError:
                raise InputError(f"'{spec['label']}' must be a number.")
            if not math.isfinite(x) or x < 0:
                raise InputError(f"'{spec['label']}' must be zero or more.")
            values[f] = x
        values["nws_hours_since_warning"] = min(values["nws_hours_since_warning"], 168.0)
        try:
            cost_ratio = float(form.get("cost_ratio", 1))
        except ValueError:
            cost_ratio = 1.0
        return state, start, values, max(cost_ratio, 0.1)

    def build_row(self, state, start, values):
        """All model features: user inputs + time features + typical values for the rest."""
        storm = 1 if values["noaa_storm_count"] > 0 else 0
        base = self.defaults[state][storm]
        row = {c: (np.nan if v is None else v) for c, v in base["num"].items()}
        row.update({c: (np.nan if v is None else v) for c, v in base["cat"].items()})
        month, hour = start.month, start.hour
        row.update(start_month=month, start_hour=hour, is_weekend=int(start.weekday() >= 5), post_regime=int(start.year >= 2011),
                   month_sin=math.sin(2 * math.pi * month / 12), month_cos=math.cos(2 * math.pi * month / 12),
                   hour_sin=math.sin(2 * math.pi * hour / 24), hour_cos=math.cos(2 * math.pi * hour / 24))
        if "regime_2024" in row:
            row["regime_2024"] = 0              # the models were trained on the pre-2024 reporting form
        for f, x in values.items():
            row[f] = math.log1p(x) if f == "hist_state_storm_cust" else x
        if storm:
            row["noaa_has_storm"] = 1
        else:
            row.update(noaa_has_storm=0, noaa_n_ongoing=0, noaa_has_ongoing_storm=0)
        return pd.DataFrame([row])[self.num + self.cat], base["n"], storm

    def calibrate(self, p):
        p = np.clip(self._iso.predict(np.atleast_1d(p)), 1e-4, 1 - 1e-4)
        a, b = p * self._pi_recent / self._pi_fit, (1 - p) * (1 - self._pi_recent) / (1 - self._pi_fit)
        return float((a / (a + b))[0])

    def predict(self, form):
        state, start, values, cost_ratio = self.parse(form)
        X, n_ref, storm = self.build_row(state, start, values)
        per_100k = math.expm1(float(self.customers_model.predict(X)[0]))
        customers = per_100k * self.population[state] / 100_000
        minutes = math.expm1(float(self.duration_model.predict(X)[0]))
        prob = self.calibrate(self.severe_model.predict_proba(X)[:, 1])
        threshold = 1 / (1 + cost_ratio)
        return {
            "state": state.title(),
            "start": start.strftime("%d %b %Y, %H:%M"),
            "customers": round(customers),
            "customers_per_100k": round(per_100k, 1),
            "duration_minutes": round(minutes),
            "duration_text": format_duration(minutes),
            "severe_probability": round(prob, 3),
            "severe": prob >= threshold,
            "severe_label": "Severe" if prob >= threshold else "Not severe",
            "threshold": round(threshold, 3),
            "cost_ratio": cost_ratio,
            "auto_filled": self.n_features - len(INPUTS) - 9,   # minus the 5 inputs and the time features
            "reference_outages": n_ref,
            "storm_status": "storm active" if storm else "no storm",
        }


def format_duration(minutes):
    minutes = max(0, int(round(minutes)))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    parts = ([f"{days} d"] if days else []) + ([f"{hours} h"] if hours else []) + ([f"{mins} min"] if mins and not days else [])
    return " ".join(parts) or "0 min"
