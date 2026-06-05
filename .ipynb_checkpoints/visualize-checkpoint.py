"""
visualize.py
============
Produces publication-quality visualizations for model comparison,
ROC curves, precision-recall curves, and feature importance.

All plots are saved under model_outputs/.

Usage:
    python visualize.py          # uses cached results_table.csv + kfold_results.csv
    python visualize.py --roc    # re-fits models to produce ROC curves (needs data)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
COLORS      = sns.color_palette("tab10")


# ── 1. Grouped bar chart ──────────────────────────────────────────────────────

def plot_grouped_bar(results_csv=OUTPUT_DIR / "results_table.csv"):
    df = pd.read_csv(results_csv)
    clfs = df["Classifier"].unique()
    feats = df["Feature Set"].unique()

    x = np.arange(len(feats))
    width = 0.25
    offsets = np.linspace(-(len(clfs) - 1) / 2, (len(clfs) - 1) / 2, len(clfs)) * width

    for metric in METRIC_COLS:
        fig, ax = plt.subplots(figsize=(max(8, len(feats) * 2), 5))
        for i, clf in enumerate(clfs):
            vals = [df[(df["Classifier"] == clf) & (df["Feature Set"] == f)][metric].values
                    for f in feats]
            vals = [v[0] if len(v) else 0 for v in vals]
            ax.bar(x + offsets[i], vals, width, label=clf, color=COLORS[i], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(feats, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Classifier and Feature Set")
        ax.set_ylim(0.5, 1.02)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        path = OUTPUT_DIR / f"bar_{metric.lower().replace('-','_').replace(' ','_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)


# ── 2. K-fold box plots ───────────────────────────────────────────────────────

def plot_kfold_boxplots(kfold_csv=None):
    """
    Plots per-fold F1 distributions.
    If kfold_csv is None, uses model_outputs/kfold_results.csv (summary only).
    For box plots we need per-fold data — call this from main.py directly.
    """
    path = kfold_csv or OUTPUT_DIR / "kfold_results.csv"
    if not Path(path).exists():
        log.warning("kfold_results.csv not found; skipping boxplot.")
        return
    df = pd.read_csv(path)
    f1 = df[df["Metric"] == "f1"].copy()
    f1["label"] = f1["Classifier"].str.split().str[0] + "\n" + f1["Feature Set"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        f1["label"],
        f1["Mean"],
        xerr=f1["Std"],
        color=sns.color_palette("husl", len(f1)),
        edgecolor="white",
        capsize=4,
    )
    ax.set_xlabel("F1-Score (mean ± std)")
    ax.set_title(f"Stratified K-Fold CV — F1 Score Summary")
    ax.axvline(0.9, color="red", linestyle="--", alpha=0.4, label="F1=0.9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kfold_f1_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("K-fold F1 summary bar chart saved.")


# ── 3. ROC curves (needs fitted models + test data) ───────────────────────────

def plot_roc_curves(models_probs: dict, y_test: np.ndarray):
    """
    Parameters
    ----------
    models_probs : dict  {label_str: y_prob_array}
    y_test       : true binary labels
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (label, probs) in enumerate(models_probs.items()):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=COLORS[i % len(COLORS)],
                label=f"{label} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curves saved.")


# ── 4. Precision-Recall curves ────────────────────────────────────────────────

def plot_pr_curves(models_probs: dict, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (label, probs) in enumerate(models_probs.items()):
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(recall, precision, lw=2, color=COLORS[i % len(COLORS)],
                label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PR curves saved.")


# ── 5. Feature importance (LR / RF) ──────────────────────────────────────────

def plot_lr_feature_importance(clf, vectorizer, top_k=20, label="LR_TFIDF"):
    coef = clf.coef_[0] if hasattr(clf, "coef_") else None
    if coef is None:
        log.warning("Classifier has no coef_ attribute; skipping LR importance plot.")
        return
    feature_names = vectorizer.get_feature_names_out()
    top_pos = np.argsort(coef)[::-1][:top_k]
    top_neg = np.argsort(coef)[:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, idx, title, color in [
        (axes[0], top_pos, "Top FAKE indicators",  "#e74c3c"),
        (axes[1], top_neg, "Top REAL indicators",  "#2ecc71"),
    ]:
        words  = [feature_names[i] for i in idx]
        values = np.abs(coef[idx])
        ax.barh(words[::-1], values[::-1], color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("|Coefficient|")
    fig.suptitle("Logistic Regression Feature Importance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"lr_importance_{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("LR importance plot saved.")


def plot_rf_feature_importance(clf, feature_names=None, top_k=20, label="RF"):
    importances = clf.feature_importances_ if hasattr(clf, "feature_importances_") else None
    if importances is None:
        log.warning("Classifier has no feature_importances_ attribute.")
        return
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    top_idx = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
        color="#3498db", edgecolor="white"
    )
    ax.set_title(f"Random Forest Top-{top_k} Feature Importances ({label})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rf_importance_{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("RF importance plot saved.")


# ── 6. Paired t-test heatmap ──────────────────────────────────────────────────

def plot_ttest_heatmap(ttest_csv=OUTPUT_DIR / "paired_ttests.csv"):
    if not Path(ttest_csv).exists():
        log.warning("paired_ttests.csv not found; skipping t-test heatmap.")
        return
    df = pd.read_csv(ttest_csv)

    models = sorted(set(df["model_a"].tolist() + df["model_b"].tolist()))
    n = len(models)
    p_mat  = np.full((n, n), np.nan)
    idx    = {m: i for i, m in enumerate(models)}

    for _, row in df.iterrows():
        i, j = idx[row["model_a"]], idx[row["model_b"]]
        p_mat[i, j] = row["p_value"]
        p_mat[j, i] = row["p_value"]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    mask = np.isnan(p_mat)
    sns.heatmap(
        p_mat, mask=mask, annot=True, fmt=".3f",
        xticklabels=models, yticklabels=models,
        cmap="coolwarm_r", vmin=0, vmax=0.1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Paired T-Test p-values (F1)")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ttest_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("T-test heatmap saved.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_grouped_bar()
    plot_kfold_boxplots()
    plot_ttest_heatmap()
    log.info("All visualizations generated in %s", OUTPUT_DIR.resolve())
