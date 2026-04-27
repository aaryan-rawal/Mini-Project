"""
pred.py — Prediction Engine
Loads trained model + vectorizer, predicts spam/ham for any text,
detects spam keywords, and logs every prediction to history.csv.
"""

import os
import pickle
import csv
import re
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "model.pkl")
VEC_PATH     = os.path.join(BASE_DIR, "model", "vec.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "history.csv")

# ── Spam Keywords ──────────────────────────────────────────────────────────
SPAM_KEYWORDS = [
    "free", "win", "winner", "won", "prize", "urgent", "congratulations",
    "claim", "offer", "click", "cash", "money", "discount", "deal",
    "limited", "exclusive", "guaranteed", "risk free", "apply now",
    "act now", "buy now", "order now", "call now", "subscribe",
    "loan", "credit", "investment", "earn", "income", "profit",
    "selected", "alert", "verify", "account", "password", "bank",
]


def load_model():
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    return mdl, vec


# Load once at module import
mdl, vec = load_model()


def detect_keywords(text):
    lower = text.lower()
    found = [kw for kw in SPAM_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lower)]
    return found


def predict(text):
    from ml.prep import clean

    cleaned  = clean(text)
    vec_text = vec.transform([cleaned])

    pred  = mdl.predict(vec_text)[0]
    label = "Spam" if pred == 1 else "Ham"

    try:
        proba = mdl.predict_proba(vec_text)[0]
        conf  = round(float(max(proba)) * 100, 2)
    except AttributeError:
        import math
        score     = mdl.decision_function(vec_text)[0]
        prob_spam = 1 / (1 + math.exp(-score))
        conf      = round((prob_spam if pred == 1 else 1 - prob_spam) * 100, 2)

    keywords = detect_keywords(text)
    log_history(text, label, conf, keywords)

    return {
        "label"     : label,
        "confidence": conf,
        "keywords"  : keywords,
    }


def log_history(text, label, conf, keywords):
    file_exists = os.path.isfile(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "message", "result", "confidence", "keywords"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text[:120],
            label,
            f"{conf}%",
            ", ".join(keywords) if keywords else "—",
        ])


def get_history(limit=20):
    if not os.path.isfile(HISTORY_PATH):
        return []
    import pandas as pd
    try:
        df = pd.read_csv(HISTORY_PATH)
        return df.tail(limit).iloc[::-1].to_dict(orient="records")
    except Exception:
        return []


if __name__ == "__main__":
    tests = [
        "Congratulations! You've WON a FREE iPhone. Click now to claim!",
        "Hey, are you coming to the study session tomorrow evening?",
    ]
    for t in tests:
        r = predict(t)
        print(f"\n[{r['label']}] {r['confidence']}% confident")
        print(f"Keywords : {r['keywords']}")
        print(f"Message  : {t[:60]}...")
