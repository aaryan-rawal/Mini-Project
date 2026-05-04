"""
eval.py — Model Evaluation Script
Loads the saved model and vectorizer, evaluates on test data,
prints all metrics, and saves the confusion matrix as an image.
"""

import os
import sys
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.prep import clean

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
VEC_PATH   = os.path.join(os.path.dirname(__file__), "..", "model", "vec.pkl")
CM_PATH    = os.path.join(os.path.dirname(__file__), "..", "confusion_matrix.png")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    return mdl, vec


def load_test_data(vec):
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # support both old (v1/v2) and new (label/message) column formats
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]
    else:
        df = df[["label", "message"]]

    df["label"]   = df["label"].map({"ham": 0, "spam": 1})
    df            = df.dropna(subset=["label", "message"])
    df["cleaned"] = df["message"].apply(clean)
    df            = df[df["cleaned"].str.strip() != ""]

    X = vec.transform(df["cleaned"])
    y = df["label"].values

    _, xv, _, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return xv, yv


def print_metrics(yv, preds):
    acc  = accuracy_score(yv, preds)
    prec = precision_score(yv, preds)
    rec  = recall_score(yv, preds)
    f1   = f1_score(yv, preds)

    print("\n" + "=" * 40)
    print("        EVALUATION METRICS")
    print("=" * 40)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print("=" * 40)
    print("\n[Classification Report]\n")
    print(classification_report(yv, preds, target_names=["Ham", "Spam"]))


def save_confusion_matrix(yv, preds):
    cm = confusion_matrix(yv, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"[SAVED] Confusion matrix → {CM_PATH}")


def run():
    print("[EVAL] Loading model and vectorizer...")
    mdl, vec = load_model()

    print("[EVAL] Preparing test data...")
    xv, yv = load_test_data(vec)

    preds = mdl.predict(xv)
    print_metrics(yv, preds)
    save_confusion_matrix(yv, preds)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    run()