"""Flask web app: early severity prediction of major power outages.

AWS Elastic Beanstalk looks for an object called `application` in application.py.
Local run:  python application.py   (then open http://127.0.0.1:8000)
"""
import os
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from predictor import INPUTS, InputError, OutagePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
application = Flask(__name__)
predictor = OutagePredictor(os.path.join(BASE_DIR, "models"))

COST_RATIOS = [(1, "1 : 1 – miss as bad as false alarm (Severe if ≥ 50%)"),
               (3, "3 : 1 – miss 3× worse (Severe if ≥ 25%)"),
               (5, "5 : 1 – miss 5× worse (Severe if ≥ 17%)")]


def page(form=None, result=None, error=None):
    form = form or {"start": datetime.now().strftime("%Y-%m-%dT%H:%M"), "cost_ratio": "1"}
    return render_template("index.html", inputs=INPUTS, states=predictor.states(), prefill=predictor.prefill(),
                           cost_ratios=COST_RATIOS, metrics=predictor.metrics, default_years=predictor.default_years,
                           n_features=predictor.n_features, form=form, result=result, error=error)


@application.get("/")
def index():
    return page()


@application.post("/")
def predict_form():
    form = request.form.to_dict()
    try:
        return page(form, result=predictor.predict(form))
    except InputError as e:
        return page(form, error=str(e)), 400


@application.post("/api/predict")
def predict_api():
    """JSON in: state, start (ISO date-time), the five inputs, optional cost_ratio. JSON out: the three predictions."""
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predictor.predict(data))
    except InputError as e:
        return jsonify({"error": str(e)}), 400


@application.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
