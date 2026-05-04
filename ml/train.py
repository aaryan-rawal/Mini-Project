"""
train.py — Model Training Script  (SpamShield v2)
"""

import os, sys, json, pickle
from datetime import datetime, timezone

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.prep import clean
from ml.data_fetcher import load_dataset, dataset_meta

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VEC_PATH   = os.path.join(MODEL_DIR, "vec.pkl")
META_PATH  = os.path.join(MODEL_DIR, "train_meta.json")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(force_remote=False):
    print("[DATA] Fetching dataset…")
    df = load_dataset(force_remote=force_remote)
    df["cleaned"] = df["message"].apply(clean)
    df = df[df["cleaned"].str.strip() != ""]   # drop empty rows
    spam_count = df["label"].sum()
    ham_count  = (df["label"] == 0).sum()
    print(f"[DATA] {len(df)} records — {spam_count} spam | {ham_count} ham")
    return df


def extract_features(df):
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,     # more features → better coverage
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    X = vec.fit_transform(df["cleaned"])
    y = df["label"].values
    return X, y, vec


def train_all(xt, yt):
    models = {
        # ComplementNB is specifically better for imbalanced text (spam datasets)
        "Complement Naive Bayes": ComplementNB(),
        # class_weight=balanced compensates for ham >> spam imbalance
        "Logistic Regression":    LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", C=5.0
        ),
        "Linear SVM":             LinearSVC(
            random_state=42, max_iter=3000, class_weight="balanced", C=1.0
        ),
    }
    for name, mdl in models.items():
        mdl.fit(xt, yt)
    return models


def evaluate_all(models, xv, yv):
    print("\n" + "=" * 52)
    print("        MODEL COMPARISON  (Test Accuracy)")
    print("=" * 52)

    best_name, best_score, best_mdl = None, 0, None
    results = {}

    for name, mdl in models.items():
        score = accuracy_score(yv, mdl.predict(xv))
        results[name] = score
        if score > best_score:
            best_score, best_name, best_mdl = score, name, mdl

    for name, score in results.items():
        tag = " ← BEST" if name == best_name else ""
        print(f"  {name:<30} : {score*100:.2f}%{tag}")

    print("-" * 52)
    print(f"  Winner: {best_name} ({best_score*100:.2f}%)")
    print("=" * 52 + "\n")
    print(classification_report(yv, best_mdl.predict(xv), target_names=["Ham", "Spam"]))
    return best_mdl, best_name, best_score, results


def save(mdl, vec, best_name, best_score, all_scores, df_meta):
    with open(MODEL_PATH, "wb") as f: pickle.dump(mdl, f)
    with open(VEC_PATH,   "wb") as f: pickle.dump(vec, f)
    meta = {
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "best_model":   best_name,
        "accuracy":     round(best_score * 100, 2),
        "all_scores":   {k: round(v * 100, 2) for k, v in all_scores.items()},
        "dataset_meta": df_meta,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] model → {MODEL_PATH}")
    print(f"[SAVED] vec   → {VEC_PATH}")


def run(force_remote=False):
    df = load_data(force_remote=force_remote)
    X, y, vec = extract_features(df)
    xt, xv, yt, yv = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train: {xt.shape[0]} | Test: {xv.shape[0]}")
    models = train_all(xt, yt)
    best, name, score, all_scores = evaluate_all(models, xv, yv)
    save(best, vec, name, score, all_scores, dataset_meta())
    print("Training complete!")
    return score


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force-remote", action="store_true")
    args = p.parse_args()
    run(force_remote=args.force_remote)
