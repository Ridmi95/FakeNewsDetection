"""
eda.py
======
Exploratory Data Analysis for the Fake News Detection project.

Produces and saves:
    eda_label_distribution.png
    eda_source_label.png
    eda_text_length.png
    eda_wordcloud_fake.png
    eda_wordcloud_real.png
    eda_top_unigrams.png
    eda_top_bigrams.png
    eda_summary.csv

Run directly:
    python eda.py
"""

import warnings
warnings.filterwarnings("ignore")

import re
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = {"fake": "#e74c3c", "real": "#2ecc71"}


# ── internal helpers ──────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _label_name(df):
    df = df.copy()
    df["label_name"] = df["label"].map({1: "fake", 0: "real"})
    return df


# ── public API ────────────────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame):
    df = _label_name(df)
    counts = df["label_name"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # bar
    colors = [PALETTE.get(l, "#aaa") for l in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Label Counts", fontsize=13)
    axes[0].set_ylabel("Count")
    # pie
    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Label Proportions", fontsize=13)
    fig.suptitle("Label Distribution", fontsize=15, fontweight="bold")
    _save(fig, "eda_label_distribution.png")


def plot_source_label(df: pd.DataFrame):
    df = _label_name(df)
    ct = pd.crosstab(df["source"], df["label_name"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ct.plot(kind="bar", ax=ax, color=[PALETTE["fake"], PALETTE["real"]], edgecolor="white")
    ax.set_title("Fake vs Real by Source", fontsize=13)
    ax.set_xlabel("Source")
    ax.set_ylabel("Count")
    ax.legend(title="Label")
    plt.xticks(rotation=0)
    _save(fig, "eda_source_label.png")


def plot_text_length(df: pd.DataFrame):
    df = _label_name(df)
    df["text_len"] = df["clean_text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axes, ["fake", "real"]):
        subset = df[df["label_name"] == label]["text_len"]
        ax.hist(subset, bins=50, color=PALETTE[label], edgecolor="white", alpha=0.85)
        ax.axvline(subset.median(), color="black", linestyle="--", label=f"median={subset.median():.0f}")
        ax.set_title(f"Text Length – {label.upper()}", fontsize=12)
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")
        ax.legend()
    fig.suptitle("Text Length Distribution", fontsize=14, fontweight="bold")
    _save(fig, "eda_text_length.png")


def plot_wordclouds(df: pd.DataFrame):
    try:
        from wordcloud import WordCloud
    except ImportError:
        log.warning("wordcloud not installed – skipping wordcloud plots.")
        return

    df = _label_name(df)
    stop_extra = {"said", "say", "says", "one", "also", "would", "could", "may", "like", "get"}

    for label in ["fake", "real"]:
        text = " ".join(df[df["label_name"] == label]["clean_text"].dropna())
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="Reds" if label == "fake" else "Greens",
            max_words=150,
            stopwords=stop_extra,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud – {label.upper()} news", fontsize=14, fontweight="bold")
        _save(fig, f"eda_wordcloud_{label}.png")


def plot_top_ngrams(df: pd.DataFrame, n=1, top_k=20):
    df = _label_name(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ngram_name = "Unigrams" if n == 1 else "Bigrams"

    for ax, label in zip(axes, ["fake", "real"]):
        corpus = df[df["label_name"] == label]["clean_text"].dropna().tolist()
        vec = CountVectorizer(ngram_range=(n, n), stop_words="english", max_features=5000)
        X = vec.fit_transform(corpus)
        freqs = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        top_idx = freqs.argsort()[::-1][:top_k]
        words = [vocab[i] for i in top_idx]
        counts = [freqs[i] for i in top_idx]

        ax.barh(words[::-1], counts[::-1], color=PALETTE[label], edgecolor="white")
        ax.set_title(f"Top {top_k} {ngram_name} – {label.upper()}", fontsize=12)
        ax.set_xlabel("Frequency")

    fig.suptitle(f"Top {ngram_name} by Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = "eda_top_unigrams.png" if n == 1 else "eda_top_bigrams.png"
    _save(fig, fname)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df = _label_name(df)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["char_count"] = df["clean_text"].str.len()

    stats = df.groupby(["source", "label_name"])[["word_count", "char_count"]].agg(
        ["count", "mean", "median", "std"]
    ).round(1)
    path = OUTPUT_DIR / "eda_summary.csv"
    stats.to_csv(path)
    log.info("Summary saved to %s", path)
    return stats


def run_full_eda(df: pd.DataFrame):
    """Run every EDA step in sequence."""
    log.info("Starting EDA …")
    plot_label_distribution(df)
    plot_source_label(df)
    plot_text_length(df)
    plot_wordclouds(df)
    plot_top_ngrams(df, n=1)
    plot_top_ngrams(df, n=2)
    stats = summary_statistics(df)
    log.info("EDA complete.  All outputs in: %s", OUTPUT_DIR.resolve())
    return stats


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    stats = run_full_eda(df)
    print("\nSummary Statistics:\n")
    print(stats.to_string())
