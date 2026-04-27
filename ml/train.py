"""
train.py — Model Training Script
Trains Naive Bayes, Logistic Regression, and Linear SVM.
Picks the best model and saves it along with the vectorizer.
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.prep import clean

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VEC_PATH   = os.path.join(MODEL_DIR, "vec.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"]   = df["label"].map({"ham": 0, "spam": 1})
    df["cleaned"] = df["message"].apply(clean)
    print(f"[DATA] Loaded {len(df)} records — {df['label'].sum()} spam, {(df['label']==0).sum()} ham")
    return df


def extract_features(df):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(df["cleaned"])
    y = df["label"].values
    return X, y, vec


def train_all(xt, yt):
    models = {
        "Naive Bayes"         : MultinomialNB(),
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM"          : LinearSVC(random_state=42, max_iter=2000),
    }
    for name, mdl in models.items():
        mdl.fit(xt, yt)
    return models


def evaluate_all(models, xv, yv):
    print("\n" + "=" * 45)
    print("       MODEL COMPARISON (Test Accuracy)")
    print("=" * 45)

    best_name  = None
    best_score = 0
    best_mdl   = None

    results = {}
    for name, mdl in models.items():
        score = accuracy_score(yv, mdl.predict(xv))
        results[name] = score
        if score > best_score:
            best_score = score
            best_name  = name
            best_mdl   = mdl

    for name, score in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<22} : {score*100:.2f}%{marker}")

    print("-" * 45)
    print(f"  Winner : {best_name} ({best_score*100:.2f}%)")
    print("=" * 45 + "\n")

    return best_mdl, best_name, best_score


def save(mdl, vec):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(mdl, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vec, f)
    print(f"[SAVED] model → {MODEL_PATH}")
    print(f"[SAVED] vec   → {VEC_PATH}")


def run():
    df = load_data()
    X, y, vec = extract_features(df)

    xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[SPLIT] Train: {xt.shape[0]} | Test: {xv.shape[0]}")

    models        = train_all(xt, yt)
    best, name, _ = evaluate_all(models, xv, yv)

    save(best, vec)
    print("Training complete!")


if __name__ == "__main__":
    run()
