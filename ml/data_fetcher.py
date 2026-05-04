"""
data_fetcher.py — Dynamic Dataset Retrieval  (SpamShield v2)
─────────────────────────────────────────────────────────────
Fetches the UCI SMS Spam Collection live from the web at runtime.
Falls back to the local CSV if the network is unavailable.

Public API
──────────
  load_dataset()  →  pd.DataFrame  with columns: label (0/1), message
  dataset_meta()  →  dict  with source, size, fetched_at, sha256
"""

import os
import io
import csv
import hashlib
import logging
import time
import zipfile
from datetime import datetime, timezone

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOCAL_CSV  = os.path.join(DATA_DIR, "spam.csv")
CACHE_CSV  = os.path.join(DATA_DIR, "spam_cache.csv")   # fetched-data cache
META_FILE  = os.path.join(DATA_DIR, ".fetch_meta.txt")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Remote sources (tried in order) ───────────────────────────────────────
# Primary: UCI ML repo zip  |  Fallback: GitHub raw mirror
SOURCES = [
    {
        "name": "UCI ML Repository",
        "url":  "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "type": "zip",
        "inner_file": "SMSSpamCollection",
    },
    {
        "name": "GitHub mirror (raw)",
        "url":  "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
        "type": "csv_kaggle",
    },
]

TIMEOUT     = 15   # seconds
MAX_RETRIES = 2


# ── Internal helpers ───────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _parse_uci_tsv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse the raw tab-separated SMSSpamCollection file."""
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rows.append(parts)
    df = pd.DataFrame(rows, columns=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df.dropna()


def _parse_kaggle_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse the Kaggle-format CSV (v1, v2, …)."""
    text = raw_bytes.decode("latin-1", errors="replace")
    df   = pd.read_csv(io.StringIO(text), usecols=[0, 1], header=0)
    df.columns = ["label", "message"]
    df["label"] = df["label"].str.strip().map({"ham": 0, "spam": 1})
    return df.dropna()


def _fetch_from_source(source: dict) -> pd.DataFrame:
    """Attempt to download and parse one source. Returns DataFrame or raises."""
    logger.info(f"[FETCH] Trying: {source['name']} → {source['url']}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(source["url"], timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            raw = resp.content
            break
        except Exception as e:
            logger.warning(f"[FETCH] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                raise

    stype = source["type"]

    if stype == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            inner = source.get("inner_file", z.namelist()[0])
            with z.open(inner) as f:
                content = f.read()
        return _parse_uci_tsv(content), raw

    elif stype == "csv_kaggle":
        return _parse_kaggle_csv(raw), raw

    else:
        raise ValueError(f"Unknown source type: {stype}")


def _save_cache(df: pd.DataFrame, sha: str, source_name: str):
    df.to_csv(CACHE_CSV, index=False)
    with open(META_FILE, "w") as f:
        f.write(f"fetched_at={datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"source={source_name}\n")
        f.write(f"sha256={sha}\n")
        f.write(f"rows={len(df)}\n")
    logger.info(f"[CACHE] Saved {len(df)} rows to {CACHE_CSV}")


def _read_meta() -> dict:
    meta = {"source": "unknown", "fetched_at": "never", "sha256": "n/a", "rows": 0}
    if os.path.isfile(META_FILE):
        with open(META_FILE) as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                meta[k] = v
    return meta


def _load_local(path: str, label: str) -> pd.DataFrame:
    logger.info(f"[LOCAL] Loading from {label}: {path}")
    df = pd.read_csv(path, encoding="latin-1")
    # handle both UCI TSV-converted and Kaggle formats
    if "v1" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]
    elif "label" not in df.columns:
        df.columns = ["label", "message"] + list(df.columns[2:])
        df = df[["label", "message"]]
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).fillna(df["label"])
    return df[["label", "message"]].dropna()


# ── Public API ─────────────────────────────────────────────────────────────

def load_dataset(force_remote: bool = False) -> pd.DataFrame:
    """
    Load the SMS Spam dataset.

    Strategy:
      1. Try each remote source in SOURCES order.
      2. On success → cache result locally and return.
      3. If all remote sources fail → use CACHE_CSV if it exists.
      4. Last resort → original local spam.csv bundled in the repo.

    Args:
      force_remote:  Skip cache and always re-fetch from network.

    Returns:
      pd.DataFrame with columns: label (int 0/1), message (str)
    """
    # Skip remote fetch if cache is fresh (< 24 h) and not forced
    if not force_remote and os.path.isfile(CACHE_CSV):
        meta = _read_meta()
        try:
            fetched_at = datetime.fromisoformat(meta.get("fetched_at", ""))
            age_hours  = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
            if age_hours < 24:
                logger.info(f"[CACHE] Using cached dataset ({age_hours:.1f}h old)")
                return _load_local(CACHE_CSV, "cache")
        except Exception:
            pass

    # Try each remote source
    for source in SOURCES:
        try:
            df, raw = _fetch_from_source(source)
            if len(df) < 500:
                raise ValueError(f"Too few rows fetched: {len(df)}")
            sha = _sha256(raw)
            _save_cache(df, sha, source["name"])
            logger.info(f"[OK] Fetched {len(df)} rows from {source['name']}")
            return df
        except Exception as e:
            logger.warning(f"[SKIP] {source['name']}: {e}")

    # Fallback → cached CSV
    if os.path.isfile(CACHE_CSV):
        logger.warning("[FALLBACK] Using stale cache")
        return _load_local(CACHE_CSV, "stale cache")

    # Last resort → bundled local CSV
    if os.path.isfile(LOCAL_CSV):
        logger.warning("[FALLBACK] Using bundled local CSV")
        return _load_local(LOCAL_CSV, "local bundle")

    raise RuntimeError("No dataset available — network unreachable and no local file found.")


def dataset_meta() -> dict:
    """Return metadata about the currently cached dataset."""
    meta = _read_meta()
    meta["cache_exists"] = os.path.isfile(CACHE_CSV)
    meta["local_exists"] = os.path.isfile(LOCAL_CSV)
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_dataset(force_remote=True)
    print(f"\nDataset shape : {df.shape}")
    print(f"Spam count    : {df['label'].sum()}")
    print(f"Ham  count    : {(df['label'] == 0).sum()}")
    print(f"\nSample:\n{df.head(3)}")
    print(f"\nMeta: {dataset_meta()}")
