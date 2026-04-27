"""
prep.py — Text Preprocessing Pipeline
Cleans and prepares SMS messages for ML training.
"""

import re
import string
import nltk

# Download required NLTK data (only needed once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Init lemmatizer and stopwords
lem = WordNetLemmatizer()
stops = set(stopwords.words("english"))


def clean(text):
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    Returns cleaned string.
    """
    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove punctuation and digits
    text = re.sub(r"[^a-z\s]", "", text)

    # Step 3 — Tokenize
    tokens = word_tokenize(text)

    # Step 4 — Remove stopwords + Step 5 — Lemmatize
    tokens = [lem.lemmatize(w) for w in tokens if w not in stops and len(w) > 1]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "FREE entry! Win a PRIZE now!!! Click here: http://win.com"
    print("Original :", sample)
    print("Cleaned  :", clean(sample))
