# Chronic Kidney Disease Detection Using Machine Learning

Educational ML project (NOT a medical diagnosis). Traditional supervised learning on the UCI CKD dataset
(Logistic Regression, Decision Tree, Random Forest, KNN, SVM), served through a Flask web page.

## Folder layout
```
CKD-ML-Project/
├── app.py                 Flask web app
├── train_model.py         Downloads data, cleans, trains 5 models, evaluates, saves model
├── model/                 model.pkl, preprocessing.pkl, features.json  (created by train_model.py)
├── results/               comparison table, confusion matrices, plots, test cases (created by train_model.py)
├── templates/index.html
├── static/style.css
├── notebook/CKD_ML_Project.ipynb   same code as train_model.py, split in cells (for Colab / viva demo)
├── requirements.txt       libraries needed by the web app (gets pinned automatically after training)
├── requirements-train.txt libraries needed for training
├── Procfile               tells the server how to start the app (gunicorn)
└── README.md
```

## A. Run locally (Windows / Mac / Linux)
1. Install Python 3.10+ and open a terminal in this folder.
2. Create a virtual environment (recommended):
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. `pip install -r requirements-train.txt`
4. `python train_model.py`  (needs internet once; takes under a minute). Read the printed tables.
   This also rewrites `requirements.txt` with the exact versions you trained with.
5. `python app.py`
6. Open http://127.0.0.1:5000 in your browser.

On Windows, `gunicorn` may fail to install/run locally. That is fine: it is only used on the server.
If `pip install` complains about gunicorn on Windows, install the rest manually:
`pip install Flask pandas numpy scikit-learn joblib matplotlib seaborn ucimlrepo`.

## B. Test cases
`train_model.py` writes `results/test_cases.csv`: 3 CKD and 3 Not-CKD rows taken from the held-out test set,
with the true label and the model's prediction. Type those values into the form.
(Leaving a field blank is allowed; at least half the fields must be filled.)
Real-world rows can be misclassified: do not expect 100% agreement.

## C. Deploy on Render (free, beginner friendly)
1. Create a free account at https://github.com and a new repository. Upload **all** project files,
   including the `model/` folder (the .pkl files must be in the repository) and `requirements.txt`.
   Do not skip `.python-version`.
2. Create a free account at https://render.com and sign in with GitHub.
3. New > Web Service > select your repository.
4. Settings: Runtime = Python 3, Build Command = `pip install -r requirements.txt`,
   Start Command = `gunicorn app:app`, Instance type = Free.
5. Click Deploy. Wait for "Your service is live".
6. Render shows your public address at the top of the service page (it ends in `.onrender.com`).
   Open it from your phone or another computer.
7. Free services sleep when idle, so the first visit after a while can take about a minute.

If the build fails with a version error, make sure the Python version in `.python-version` is supported by Render
(otherwise set the environment variable PYTHON_VERSION in Render to a supported version and re-train locally with it).
To update the site later, push changes to GitHub; Render redeploys automatically.

## Important
The model is trained on a small dataset (a few hundred patients from one hospital region). Results do not
imply clinical accuracy. Use only for education.
