"""
data_loader.py
==============
Loads and preprocesses the FakeNewsNet dataset (PolitiFact + GossipCop).
Expects four CSV files in the same directory (or a configurable DATA_DIR):
    politifact_fake.csv, politifact_real.csv
    gossipcop_fake.csv,  gossipcop_real.csv

Each CSV is expected to have at minimum a text column (auto-detected from
common column names: 'title', 'text', 'content', 'news_url').
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── configurable paths ────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("FAKENEWS_DATA_DIR", "./Dataset"))

FILE_MAP = {
    "politifact": {
        "fake": DATA_DIR / "politifact_fake.csv",
        "real": DATA_DIR / "politifact_real.csv",
    },
    "gossipcop": {
        "fake": DATA_DIR / "gossipcop_fake.csv",
        "real": DATA_DIR / "gossipcop_real.csv",
    },
}

# Candidate column names for the main text (tried in order)
TEXT_CANDIDATES = ["text", "content", "title", "news_url", "statement"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col
    # fallback: largest-average-length string column
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        raise ValueError("No string columns found in dataframe")
    avg_lens = {c: df[c].dropna().str.len().mean() for c in str_cols}
    return max(avg_lens, key=avg_lens.get)


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(sources=("politifact", "gossipcop")) -> pd.DataFrame:
    """
    Load and merge dataset files.

    Parameters
    ----------
    sources : iterable of str
        Which source(s) to include.  Options: 'politifact', 'gossipcop'.

    Returns
    -------
    pd.DataFrame
        Columns: id, source, text, label (0 = real, 1 = fake), clean_text
    """
    frames = []
    for src in sources:
        if src not in FILE_MAP:
            raise ValueError(f"Unknown source '{src}'. Choose from {list(FILE_MAP)}")
        for label_name, path in FILE_MAP[src].items():
            if not path.exists():
                log.warning("File not found, skipping: %s", path)
                continue
            df = pd.read_csv(path, low_memory=False)
            log.info("Loaded %s rows from %s", len(df), path.name)
            text_col = _detect_text_column(df)
            df = df.rename(columns={text_col: "text"})
            df["source"] = src
            df["label"] = 1 if label_name == "fake" else 0
            # preserve an id column if available
            if "id" not in df.columns:
                df["id"] = range(len(df))
            frames.append(df[["id", "source", "text", "label"]])

    if not frames:
        raise FileNotFoundError(
            "No CSV files were found. Set FAKENEWS_DATA_DIR or place the "
            "four CSVs next to this script."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["text"], inplace=True)
    combined.dropna(subset=["text"], inplace=True)
    combined["clean_text"] = combined["text"].apply(_clean_text)
    combined = combined[combined["clean_text"].str.len() > 10].reset_index(drop=True)

    log.info(
        "Combined dataset: %d rows | fake=%d | real=%d",
        len(combined),
        combined["label"].sum(),
        (combined["label"] == 0).sum(),
    )
    return combined


def get_X_y(sources=("politifact", "gossipcop")):
    """Convenience wrapper — returns (X_text, y) arrays."""
    df = load_dataset(sources)
    return df["clean_text"].values, df["label"].values, df


if __name__ == "__main__":
    X, y, df = get_X_y()
    print(df.head())
    print("\nShape:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())
