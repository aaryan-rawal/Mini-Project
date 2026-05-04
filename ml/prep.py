"""
prep.py — Advanced NLP Preprocessing Pipeline  (SpamShield v2)
──────────────────────────────────────────────────────────────
Upgrades over v1:
  • URLs, phones, currency → semantic tokens (__url__ __phone__ __money__)
    instead of stripping — these are strong spam signals.
  • POS-aware lemmatisation (verbs lemmatised as verbs, not nouns).
  • Spam-critical words (free, win, prize…) never removed as stopwords.
  • extract_nlp_features() for explainability in pred.py UI.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# ── Auto-download required NLTK assets once ───────────────────────────────
_NEEDED = {
    "tokenizers": ["punkt", "punkt_tab"],
    "corpora":    ["stopwords", "wordnet", "omw-1.4"],
    "taggers":    ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"],
}

def _ensure():
    for category, pkgs in _NEEDED.items():
        for pkg in pkgs:
            try:
                nltk.data.find(f"{category}/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)

_ensure()

# ── Singleton resources ────────────────────────────────────────────────────
_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stops      = set(stopwords.words("english"))

# Spam-signal words that look like stopwords — keep them
_SPAM_SIGNALS = {
    "free", "win", "winner", "won", "prize", "cash", "money", "earn",
    "urgent", "claim", "offer", "call", "now", "click", "buy", "order",
    "apply", "limited", "exclusive", "guaranteed", "selected", "alert",
    "verify", "credit", "loan", "risk", "act", "subscribe",
}
_EFFECTIVE_STOPS = _stops - _SPAM_SIGNALS

# ── Regex patterns ─────────────────────────────────────────────────────────
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.I)
_PHONE_RE   = re.compile(r"\b(\+?[\d][\d\s\-\(\)]{6,}\d)\b")
_MONEY_RE   = re.compile(r"[£$€¥]\s?\d[\d,\.]*|\b\d[\d,\.]*\s?(?:usd|gbp|eur|dollars?|pounds?)\b", re.I)
_NUM_RE     = re.compile(r"\b\d+\b")
_MULTI_EXCL = re.compile(r"!{2,}")
_MULTI_QUES = re.compile(r"\?{2,}")


def _normalise(text):
    text = _URL_RE.sub(" __url__ ", text)
    text = _MONEY_RE.sub(" __money__ ", text)
    text = _PHONE_RE.sub(" __phone__ ", text)
    text = _MULTI_EXCL.sub(" __multiexcl__ ", text)
    text = _MULTI_QUES.sub(" __multiques__ ", text)
    text = _NUM_RE.sub(" __num__ ", text)
    return text


def _penn_to_wordnet(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def clean(text, use_stemming=False):
    """
    Full NLP pipeline:
      1. Lowercase
      2. Normalise URLs / phones / money / repeated punctuation → tokens
      3. Strip remaining punctuation (underscores kept for placeholders)
      4. NLTK word tokenisation
      5. Remove non-spam stopwords
      6. POS-aware lemmatisation  (+optional stemming)
    Returns a whitespace-joined cleaned string for TF-IDF.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _normalise(text)
    text = re.sub(r"[^\w\s]", " ", text)  # keep underscores (in placeholders)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    processed = []
    for word, tag in tagged:
        if word in _EFFECTIVE_STOPS:
            continue
        if len(word) < 2 and not word.startswith("__"):
            continue
        wn_pos = _penn_to_wordnet(tag)
        lemma  = _lemmatizer.lemmatize(word, pos=wn_pos)
        if use_stemming:
            lemma = _stemmer.stem(lemma)
        processed.append(lemma)

    return " ".join(processed)


def extract_nlp_features(text):
    """Hand-crafted NLP feature dict for UI explainability layer."""
    raw   = text.lower()
    words = text.split()
    upper = sum(1 for c in text if c.isupper())
    return {
        "char_count":      len(text),
        "word_count":      len(words),
        "url_count":       len(_URL_RE.findall(text)),
        "phone_count":     len(_PHONE_RE.findall(text)),
        "money_count":     len(_MONEY_RE.findall(text)),
        "exclamation":     text.count("!"),
        "question_marks":  text.count("?"),
        "uppercase_ratio": round(upper / max(len(text), 1), 3),
        "avg_word_len":    round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "has_free":        int("free" in raw),
        "has_win":         int(bool(re.search(r"\bwin(ner)?\b", raw))),
        "has_urgent":      int(bool(re.search(r"\burgent\b|\bimportant\b", raw))),
        "has_claim":       int(bool(re.search(r"\bclaim\b|\bprize\b", raw))),
        "has_url":         int(bool(_URL_RE.search(text))),
        "has_phone":       int(bool(_PHONE_RE.search(text))),
        "has_money":       int(bool(_MONEY_RE.search(text))),
    }


if __name__ == "__main__":
    samples = [
        "FREE entry! Win a PRIZE now!!! Click here: http://win.com",
        "Hey, are you coming to the study session at 7pm tonight?",
        "URGENT: Your account will be SUSPENDED. Call +447700900123 NOW.",
        "Congratulations! You have won £900. Claim within 12 hours!",
    ]
    for s in samples:
        print(f"\nOriginal : {s}")
        print(f"Cleaned  : {clean(s)}")
        print(f"Features : {extract_nlp_features(s)}")
