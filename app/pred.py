"""
pred.py — Prediction Engine  (SpamShield v2)
"""

import os, sys, json, csv, math, pickle, re, threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.prep import clean

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "model.pkl")
VEC_PATH     = os.path.join(BASE_DIR, "model", "vec.pkl")
META_PATH    = os.path.join(BASE_DIR, "model", "train_meta.json")
HISTORY_PATH = os.path.join(BASE_DIR, "history.csv")

SPAM_KEYWORDS = [
    "free", "win", "winner", "won", "prize", "urgent", "congratulations",
    "claim", "offer", "click", "cash", "money", "discount", "deal",
    "limited", "exclusive", "guaranteed", "risk free", "apply now",
    "act now", "buy now", "order now", "call now", "subscribe",
    "loan", "credit", "investment", "earn", "income", "profit",
    "selected", "alert", "verify", "account", "password", "bank",
]

_lock      = threading.Lock()
_model_ref = {"mdl": None, "vec": None, "meta": {}}


def _load_model():
    with open(MODEL_PATH, "rb") as f: mdl = pickle.load(f)
    with open(VEC_PATH,   "rb") as f: vec = pickle.load(f)
    meta = {}
    if os.path.isfile(META_PATH):
        with open(META_PATH) as f: meta = json.load(f)
    return mdl, vec, meta


def _get_model():
    with _lock:
        if _model_ref["mdl"] is None:
            mdl, vec, meta     = _load_model()
            _model_ref["mdl"]  = mdl
            _model_ref["vec"]  = vec
            _model_ref["meta"] = meta
    return _model_ref["mdl"], _model_ref["vec"], _model_ref["meta"]


def reload_model():
    with _lock:
        mdl, vec, meta     = _load_model()
        _model_ref["mdl"]  = mdl
        _model_ref["vec"]  = vec
        _model_ref["meta"] = meta
    return _model_ref["meta"]


def detect_keywords(text):
    lower = text.lower()
    return [kw for kw in SPAM_KEYWORDS
            if re.search(r"\b" + re.escape(kw) + r"\b", lower)]


def predict(text):
    mdl, vec, train_meta = _get_model()

    cleaned  = clean(text)
    vec_text = vec.transform([cleaned])
    pred     = mdl.predict(vec_text)[0]
    label    = "Spam" if pred == 1 else "Ham"

    try:
        proba = mdl.predict_proba(vec_text)[0]
        conf  = round(float(max(proba)) * 100, 2)
    except AttributeError:
        score     = mdl.decision_function(vec_text)[0]
        prob_spam = 1 / (1 + math.exp(-score))
        conf      = round((prob_spam if pred == 1 else 1 - prob_spam) * 100, 2)

    keywords = detect_keywords(text)
    log_history(text, label, conf, keywords)

    return {
        "label":      label,
        "confidence": conf,
        "keywords":   keywords,
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


def get_model_info():
    _, _, meta = _get_model()
    return meta
