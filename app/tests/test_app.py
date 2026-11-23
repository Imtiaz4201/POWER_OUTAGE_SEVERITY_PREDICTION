"""Smoke test. Run from the app folder:  python tests/test_app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from application import application, predictor  # noqa: E402


def main():
    c = application.test_client()
    assert c.get("/health").json == {"status": "ok"}
    assert c.get("/").status_code == 200
    state = "TEXAS" if "TEXAS" in predictor.defaults else next(iter(predictor.defaults))
    typical = predictor.prefill()[state][1]
    body = {"state": state, "start": "2023-06-15T14:30", "cost_ratio": 3, **typical}
    r = c.post("/api/predict", json=body)
    assert r.status_code == 200, r.json
    out = r.json
    assert out["customers"] >= 0 and out["duration_minutes"] >= 0 and 0 <= out["severe_probability"] <= 1
    print("API:", {k: out[k] for k in ("state", "customers", "duration_text", "severe_probability", "severe_label")})
    assert c.post("/api/predict", json={"state": "NOWHERE"}).status_code == 400
    assert c.post("/", data={**{k: str(v) for k, v in body.items()}}).status_code == 200
    print("all tests passed")


if __name__ == "__main__":
    main()
