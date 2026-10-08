"""Flask web app for CKD prediction (educational project, NOT a medical diagnosis)."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

app = Flask(__name__)

# ---------- Load the artifacts saved by train_model.py ----------
LOAD_ERROR = None
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    preprocessing = joblib.load(MODEL_DIR / "preprocessing.pkl")
    FEATURE_INFO = json.loads((MODEL_DIR / "features.json").read_text())
    NUMERIC = FEATURE_INFO["numeric"]
    CATEGORICAL = FEATURE_INFO["categorical"]
    CATEGORIES = FEATURE_INFO["categories"]
    ALL_FEATURES = NUMERIC + CATEGORICAL
except Exception as err:  # shown on the page instead of crashing
    LOAD_ERROR = f"Model files could not be loaded: {err}. Run train_model.py first."
    model = preprocessing = None
    FEATURE_INFO, NUMERIC, CATEGORICAL, CATEGORIES, ALL_FEATURES = {}, [], [], {}, []

# ---------- Display information for each form field ----------
# min/max are generous sanity limits for typing mistakes, not medical reference ranges.
FIELDS = {
    "age": {"label": "Age", "unit": "years", "min": 1, "max": 120, "step": 1},
    "bp": {"label": "Blood Pressure", "unit": "mm Hg", "min": 30, "max": 250, "step": 1},
    "sg": {"label": "Urine Specific Gravity", "choices": ["1.005", "1.010", "1.015", "1.020", "1.025"]},
    "al": {"label": "Albumin (urine, 0-5)", "choices": ["0", "1", "2", "3", "4", "5"]},
    "su": {"label": "Sugar (urine, 0-5)", "choices": ["0", "1", "2", "3", "4", "5"]},
    "bgr": {"label": "Blood Glucose (random)", "unit": "mg/dL", "min": 10, "max": 900, "step": 1},
    "bu": {"label": "Blood Urea", "unit": "mg/dL", "min": 1, "max": 500, "step": 0.1},
    "sc": {"label": "Serum Creatinine", "unit": "mg/dL", "min": 0.1, "max": 100, "step": 0.1},
    "sod": {"label": "Sodium", "unit": "mEq/L", "min": 1, "max": 200, "step": 0.1},
    "pot": {"label": "Potassium", "unit": "mEq/L", "min": 1, "max": 60, "step": 0.1},
    "hemo": {"label": "Hemoglobin", "unit": "g/dL", "min": 1, "max": 25, "step": 0.1},
    "pcv": {"label": "Packed Cell Volume", "unit": "%", "min": 5, "max": 80, "step": 1},
    "wbcc": {"label": "White Blood Cell Count", "unit": "cells/cumm", "min": 500, "max": 50000, "step": 100},
    "rbcc": {"label": "Red Blood Cell Count", "unit": "millions/cmm", "min": 1, "max": 10, "step": 0.1},
    "rbc": {"label": "Red Blood Cells (urine)"},
    "pc": {"label": "Pus Cell"},
    "pcc": {"label": "Pus Cell Clumps"},
    "ba": {"label": "Bacteria"},
    "htn": {"label": "Hypertension"},
    "dm": {"label": "Diabetes Mellitus"},
    "cad": {"label": "Coronary Artery Disease"},
    "appet": {"label": "Appetite"},
    "pe": {"label": "Pedal Edema"},
    "ane": {"label": "Anemia"},
}

GROUPS = [
    ("Patient & Vitals", ["age", "bp"]),
    ("Urine Tests", ["sg", "al", "su", "rbc", "pc", "pcc", "ba"]),
    ("Blood Tests", ["bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]),
    ("Medical History & Symptoms", ["htn", "dm", "cad", "appet", "pe", "ane"]),
]


def build_form_groups():
    """Build the form layout using exactly the features the trained model needs."""
    groups, used = [], set()
    for title, cols in GROUPS:
        items = [make_field(c) for c in cols if c in ALL_FEATURES]
        used.update(cols)
        if items:
            groups.append({"title": title, "fields": items})
    extra = [make_field(c) for c in ALL_FEATURES if c not in used]
    if extra:
        groups.append({"title": "Other", "fields": extra})
    return groups


def make_field(col):
    info = dict(FIELDS.get(col, {"label": col}))
    info["name"] = col
    if col in CATEGORICAL:
        info["choices"] = CATEGORIES[col]
        info["kind"] = "select"
    elif "choices" in info:
        info["kind"] = "select"
    else:
        info["kind"] = "number"
    return info


def parse_form(form):
    """Turn the submitted form into a one-row DataFrame in the training column order."""
    row, errors, entered = {}, [], {}
    for col in ALL_FEATURES:
        label = FIELDS.get(col, {}).get("label", col)
        raw = (form.get(col) or "").strip()
        entered[col] = raw
        if raw == "":
            row[col] = np.nan  # blank = unknown; the saved pipeline imputes it
            continue
        if col in CATEGORICAL:
            if raw not in CATEGORIES[col]:
                errors.append(f"{label}: invalid choice.")
            else:
                row[col] = raw
        else:
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"{label}: please enter a number.")
                continue
            info = FIELDS.get(col, {})
            if "min" in info and not (info["min"] <= value <= info["max"]):
                errors.append(f"{label}: value should be between {info['min']} and {info['max']}.")
                continue
            if "choices" in info and raw not in info["choices"]:
                errors.append(f"{label}: invalid choice.")
                continue
            row[col] = value

    filled = sum(1 for v in row.values() if not (isinstance(v, float) and np.isnan(v)))
    min_filled = max(1, len(ALL_FEATURES) // 2)
    if not errors and filled < min_filled:
        errors.append(f"Please fill in at least {min_filled} of the {len(ALL_FEATURES)} fields "
                      f"(you filled {filled}). Blank fields are filled with typical training values.")

    frame = pd.DataFrame([row], columns=ALL_FEATURES)
    frame[CATEGORICAL] = frame[CATEGORICAL].astype(object)
    return frame, errors, entered


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, entered = None, [], {}
    if LOAD_ERROR:
        errors = [LOAD_ERROR]
    elif request.method == "POST":
        frame, errors, entered = parse_form(request.form)
        if not errors:
            processed = preprocessing.transform(frame)  # same preprocessing as training
            prediction = int(model.predict(processed)[0])
            result = {"positive": prediction == 1,
                      "text": "CKD Detected" if prediction == 1 else "CKD Not Detected"}
    return render_template("index.html", groups=build_form_groups(), result=result, errors=errors,
                           entered=entered, info=FEATURE_INFO)


@app.route("/health")
def health():
    return {"status": "ok" if not LOAD_ERROR else "model not loaded"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
