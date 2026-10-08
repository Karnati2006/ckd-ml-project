"""
Chronic Kidney Disease (CKD) - training script.

Run from the project folder:
    python train_model.py              (downloads the UCI dataset automatically)
    python train_model.py --csv my.csv (use your own CSV copy instead)

The same file is converted into notebook/CKD_ML_Project.ipynb (Colab).
Sections are separated by lines that start with "# %%".
"""

# %% Step 0 - Imports and settings
import argparse
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_DIR = BASE / "model"
RESULTS_DIR = BASE / "results"
DATA_DIR = BASE / "data"
for folder in (MODEL_DIR, RESULTS_DIR, DATA_DIR):
    folder.mkdir(exist_ok=True)

IN_NOTEBOOK = "ipykernel" in sys.modules

NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
           "pot", "hemo", "pcv", "wbcc", "rbcc"]
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
TARGET = "class"

VALID_CATEGORIES = {
    "rbc": ["abnormal", "normal"],
    "pc": ["abnormal", "normal"],
    "pcc": ["notpresent", "present"],
    "ba": ["notpresent", "present"],
    "htn": ["no", "yes"],
    "dm": ["no", "yes"],
    "cad": ["no", "yes"],
    "appet": ["good", "poor"],
    "pe": ["no", "yes"],
    "ane": ["no", "yes"],
}
# Column names used by some other copies of this dataset (e.g. Kaggle)
RENAME = {"classification": "class", "wc": "wbcc", "rc": "rbcc"}

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=None, help="optional path to a CKD csv file")
args, _ = parser.parse_known_args()


def finish_plot(filename):
    """Save the current figure into results/ and show it if in a notebook."""
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


# %% Step 1 - Load the dataset
def load_raw_data(csv_path):
    if csv_path:
        print(f"Reading CSV: {csv_path}")
        return pd.read_csv(csv_path)
    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=336)  # UCI Chronic Kidney Disease
        raw_df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        raw_df.to_csv(DATA_DIR / "ckd_raw.csv", index=False)
        print("Downloaded from UCI and cached in data/ckd_raw.csv")
        return raw_df
    except Exception as err:
        cached = DATA_DIR / "ckd_raw.csv"
        if cached.exists():
            print(f"Download failed ({err}). Using cached data/ckd_raw.csv")
            return pd.read_csv(cached)
        raise RuntimeError(
            "Could not download the dataset and no local copy exists.\n"
            "Download the CKD csv manually and run: python train_model.py --csv path/to/file.csv\n"
            f"Original error: {err}"
        )


raw = load_raw_data(args.csv)
print("Raw dataset shape (rows, columns):", raw.shape)

# %% Step 2 - Understand the raw data
print(raw.head())
print("\n--- info ---")
raw.info()
print("\n--- describe (numeric) ---")
print(raw.describe().T)
print("\n--- missing values per column (raw) ---")
print(raw.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows (raw):", raw.duplicated().sum())

_target_col = "class" if "class" in raw.columns else "classification"
print("\nRaw target values:")
print(raw[_target_col].value_counts(dropna=False))


# %% Step 3 - Clean the data (no statistics are learned here)
def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=RENAME)
    df = df.drop(columns=[c for c in ["id"] if c in df.columns])

    needed = NUMERIC + CATEGORICAL + [TARGET]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"These expected columns are missing from the data: {missing_cols}\n"
                         f"Columns found: {list(df.columns)}")
    df = df[needed]

    # Text columns: remove stray spaces/tabs, lower-case, mark junk as missing
    for col in CATEGORICAL + [TARGET]:
        s = df[col].astype(str).str.strip().str.lower()
        s = s.replace({"?": np.nan, "": np.nan, "nan": np.nan, "none": np.nan})
        valid = VALID_CATEGORIES.get(col, ["ckd", "notckd"])
        s = s.where(s.isin(valid), np.nan)  # anything unexpected -> missing
        df[col] = s.astype(object)

    # Numeric columns: convert to numbers, impossible negatives -> missing
    for col in NUMERIC:
        s = df[col].astype(str).str.strip().replace({"?": np.nan, "": np.nan})
        s = pd.to_numeric(s, errors="coerce")
        df[col] = s.mask(s < 0)

    # Target: rows without a label cannot be used
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"Rows dropped because target was missing: {before - len(df)}")
    df[TARGET] = df[TARGET].map({"notckd": 0, "ckd": 1}).astype(int)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {before - len(df)}")
    return df


df = clean_data(raw)
print("\nCleaned shape:", df.shape)
print("\nMissing values after cleaning (these are imputed later, INSIDE the pipeline):")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nTarget distribution (1 = CKD, 0 = Not CKD):")
print(df[TARGET].value_counts())

# %% Step 4 - Exploratory data analysis
eda = df.assign(diagnosis=df[TARGET].map({1: "CKD", 0: "Not CKD"}))

plt.figure(figsize=(5, 4))
sns.countplot(data=eda, x="diagnosis", order=["CKD", "Not CKD"])
plt.title("CKD vs Not CKD")
finish_plot("target_distribution.png")

key_numeric = [c for c in ["age", "bgr", "bu", "sc", "sod", "hemo", "pcv", "rbcc"] if c in df.columns]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), key_numeric):
    sns.histplot(data=eda, x=col, hue="diagnosis", kde=True, ax=ax, stat="density", common_norm=False)
    ax.set_title(col)
finish_plot("numeric_distributions.png")

plt.figure(figsize=(10, 8))
sns.heatmap(df[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation (numeric features + target)")
finish_plot("correlation_heatmap.png")

# %% Step 5 - Train/test split (BEFORE any fitting) and the shared preprocessing
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print("Train size:", X_train.shape, " Test size:", X_test.shape)


def build_preprocessor():
    numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())])
    categorical_steps = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_steps, NUMERIC),
                              ("cat", categorical_steps, CATEGORICAL)])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
}


def make_pipeline(classifier):
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])


# %% Step 5b - Cross-validation on the TRAINING set only (used to choose the best model)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, clf in MODELS.items():
    scores = cross_validate(make_pipeline(clf), X_train, y_train, cv=cv,
                            scoring=["accuracy", "precision", "recall", "f1"])
    cv_rows.append({"Model": name,
                    "CV Accuracy": scores["test_accuracy"].mean(),
                    "CV Precision": scores["test_precision"].mean(),
                    "CV Recall": scores["test_recall"].mean(),
                    "CV F1": scores["test_f1"].mean()})
cv_table = pd.DataFrame(cv_rows).set_index("Model")
print("5-fold cross-validation on the training set:")
print(cv_table.round(4))

# %% Step 6 - Fit on train, evaluate on the untouched test set
fitted = {}
test_rows = []
predictions = {}
for name, clf in MODELS.items():
    pipe = make_pipeline(clf).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted[name] = pipe
    predictions[name] = pred
    test_rows.append({"Model": name,
                      "Accuracy": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred, zero_division=0),  # CKD = positive class
                      "Recall": recall_score(y_test, pred, zero_division=0),
                      "F1 Score": f1_score(y_test, pred, zero_division=0)})
test_table = pd.DataFrame(test_rows).set_index("Model")
print("Test-set results (positive class = CKD):")
print(test_table.round(4))
print(f"\nTest set has only {len(y_test)} patients, so each wrong prediction moves the numbers a lot.")

fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 4))
for ax, name in zip(axes, MODELS):
    ConfusionMatrixDisplay.from_predictions(y_test, predictions[name], display_labels=["Not CKD", "CKD"],
                                            ax=ax, colorbar=False)
    ax.set_title(name)
finish_plot("confusion_matrices.png")

comparison = cv_table.join(test_table)
comparison.round(4).to_csv(RESULTS_DIR / "model_comparison.csv")

# %% Step 6b - Choose the best model
# Rule: highest cross-validated F1 on the TRAINING data; ties are broken by CV recall
# (for a screening tool, missing a CKD patient is the costlier error).
# The test set is NOT used to choose, so the reported test numbers stay honest.
ranking = cv_table.sort_values(["CV F1", "CV Recall"], ascending=False)
best_name = ranking.index[0]
tied = ranking[(ranking["CV F1"].round(6) == ranking.iloc[0]["CV F1"].round(6)) &
               (ranking["CV Recall"].round(6) == ranking.iloc[0]["CV Recall"].round(6))].index.tolist()
print("Models ranked by CV F1 then CV recall:")
print(ranking[["CV F1", "CV Recall"]].round(4))
print(f"\nSelected model: {best_name}")
if len(tied) > 1:
    print(f"NOTE: {tied} are tied on both criteria; the first listed was kept. "
          "Say this honestly in your report - the data cannot separate them.")

# %% Step 7 - Save the model, preprocessing and feature information
best_pipe = fitted[best_name]
joblib.dump(best_pipe.named_steps["prep"], MODEL_DIR / "preprocessing.pkl")
joblib.dump(best_pipe.named_steps["clf"], MODEL_DIR / "model.pkl")

feature_info = {
    "numeric": NUMERIC,
    "categorical": CATEGORICAL,
    "categories": VALID_CATEGORIES,
    "numeric_stats": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                          "median": float(X_train[c].median())} for c in NUMERIC},
    "target_meaning": {"0": "Not CKD", "1": "CKD"},
    "best_model": best_name,
    "test_metrics": {k: float(v) for k, v in test_table.loc[best_name].items()},
    "sklearn_version": md.version("scikit-learn"),
    "python_version": platform.python_version(),
}
(MODEL_DIR / "features.json").write_text(json.dumps(feature_info, indent=2))
print("Saved: model/model.pkl, model/preprocessing.pkl, model/features.json")

# %% Step 7b - Test cases from REAL held-out rows + pin library versions
cases = X_test.copy()
cases["n_missing"] = cases.isna().sum(axis=1)
cases["true_label"] = y_test.map({1: "CKD", 0: "Not CKD"})
cases["model_prediction"] = pd.Series(best_pipe.predict(X_test), index=X_test.index).map({1: "CKD", 0: "Not CKD"})
cases = cases.sort_values("n_missing", kind="stable")
picked = pd.concat([cases[cases.true_label == "CKD"].head(3), cases[cases.true_label == "Not CKD"].head(3)])
picked.to_csv(RESULTS_DIR / "test_cases.csv", index=False)
print("\nTest cases saved to results/test_cases.csv (real rows from the test split):")
print(picked.to_string())

if not IN_NOTEBOOK:
    pins = [f"{p}=={md.version(p)}" for p in ["Flask", "pandas", "numpy", "scikit-learn", "joblib"]]
    (BASE / "requirements.txt").write_text("\n".join(pins + ["gunicorn"]) + "\n")
    (BASE / ".python-version").write_text(platform.python_version() + "\n")
    print("\nrequirements.txt and .python-version updated to match this training environment.")
