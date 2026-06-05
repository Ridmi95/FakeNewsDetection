"""
models.py
=========
Classifier training, evaluation, stratified k-fold validation,
and paired t-tests.

Classifiers:
    - Logistic Regression
    - Random Forest
    - SVM (LinearSVC / SVC with probability)

Evaluation metrics:
    accuracy, precision, recall, F1, ROC-AUC

Statistical testing:
    Paired t-test across k-fold scores

Usage:
    python models.py
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from scipy.sparse import issparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


# ── classifier definitions ────────────────────────────────────────────────────

def get_classifiers():
    """Return dict of name -> sklearn estimator (all support predict_proba)."""
    lr = LogisticRegression(
        C=1.0, max_iter=1000, solver="saga",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1,
        random_state=RANDOM_STATE
    )
    svm_base = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    svm = CalibratedClassifierCV(svm_base, cv=3)   # wraps for predict_proba

    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "SVM (LinearSVC)": svm,
    }


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_prob=None, label="") -> dict:
    """
    Compute accuracy, precision, recall, F1, ROC-AUC.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted labels
    y_prob  : predicted probabilities for positive class (for ROC-AUC)
    label   : descriptive string for logging

    Returns
    -------
    dict of metric name -> float
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")

    if label:
        log.info(
            "[%s] acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            label,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1"], metrics["roc_auc"],
        )
    return metrics


# ── single hold-out evaluation ────────────────────────────────────────────────

def train_and_evaluate(
    clf,
    X_train, y_train,
    X_test,  y_test,
    clf_name="model",
    feature_name="features",
    plot_cm=True,
):
    """
    Fit classifier, evaluate on test set, optionally plot confusion matrix.

    Returns
    -------
    dict of metrics
    """
    log.info("Training %s on %s …", clf_name, feature_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob, label=f"{clf_name}/{feature_name}")

    if plot_cm:
        _plot_confusion_matrix(y_test, y_pred, clf_name, feature_name)

    return metrics


def _plot_confusion_matrix(y_true, y_pred, clf_name, feat_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["real", "fake"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{clf_name}\n{feat_name}", fontsize=11)
    fname = f"cm_{clf_name.replace(' ', '_')}_{feat_name.replace(' ', '_')}.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved: %s", path)


# ── stratified k-fold ─────────────────────────────────────────────────────────

def stratified_kfold_cv(clf, X, y, n_splits=N_FOLDS, clf_name="model"):
    """
    Run stratified k-fold cross-validation.

    Returns
    -------
    dict of metric -> array of fold scores
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {m: [] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        if issparse(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
        else:
            X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None

        m = evaluate(y_te, y_pred, y_prob)
        for k, v in m.items():
            fold_metrics[k].append(v)

    summary = {k: np.array(v) for k, v in fold_metrics.items()}
    means = {k: v.mean() for k, v in summary.items()}
    stds  = {k: v.std()  for k, v in summary.items()}
    log.info(
        "[%s | %d-fold CV] acc=%.4f±%.4f  f1=%.4f±%.4f  auc=%.4f±%.4f",
        clf_name, n_splits,
        means["accuracy"], stds["accuracy"],
        means["f1"],       stds["f1"],
        means["roc_auc"],  stds["roc_auc"],
    )
    return summary   # dict metric -> (n_splits,) array


# ── paired t-test ─────────────────────────────────────────────────────────────

def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray,
                 name_a="A", name_b="B", metric="f1", alpha=0.05) -> dict:
    """
    Perform a paired (two-sided) t-test between two k-fold score arrays.

    Parameters
    ----------
    scores_a, scores_b : 1-D arrays of length k (per-fold scores)
    alpha              : significance level

    Returns
    -------
    dict with keys: t_stat, p_value, significant, mean_diff
    """
    assert len(scores_a) == len(scores_b), "Fold count mismatch"
    t, p = stats.ttest_rel(scores_a, scores_b)
    diff = scores_a.mean() - scores_b.mean()
    result = dict(
        metric=metric,
        model_a=name_a,
        model_b=name_b,
        mean_a=scores_a.mean(),
        mean_b=scores_b.mean(),
        mean_diff=diff,
        t_stat=t,
        p_value=p,
        significant=bool(p < alpha),
        alpha=alpha,
    )
    winner = name_a if diff > 0 else name_b
    log.info(
        "Paired t-test (%s): %s vs %s | t=%.4f  p=%.4f  sig=%s  better=%s",
        metric, name_a, name_b, t, p, result["significant"], winner,
    )
    return result


def run_all_paired_ttests(kfold_results: dict, metric="f1", alpha=0.05) -> pd.DataFrame:
    """
    Run all pairwise paired t-tests between classifiers.

    Parameters
    ----------
    kfold_results : dict  {clf_name -> {metric -> fold_scores_array}}

    Returns
    -------
    pd.DataFrame
    """
    names = list(kfold_results.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            scores_a = kfold_results[a][metric]
            scores_b = kfold_results[b][metric]
            rows.append(paired_ttest(scores_a, scores_b, a, b, metric, alpha))
    return pd.DataFrame(rows)


# ── aggregated results table ──────────────────────────────────────────────────

def build_results_table(all_results: dict) -> pd.DataFrame:
    """
    Build a tidy results table from nested dict:
        all_results[feature_name][clf_name] = metrics_dict

    Returns pd.DataFrame.
    """
    rows = []
    for feat_name, clf_dict in all_results.items():
        for clf_name, m in clf_dict.items():
            rows.append({
                "Feature Set":  feat_name,
                "Classifier":   clf_name,
                "Accuracy":     round(m["accuracy"],  4),
                "Precision":    round(m["precision"], 4),
                "Recall":       round(m["recall"],    4),
                "F1-Score":     round(m["f1"],        4),
                "ROC-AUC":      round(m["roc_auc"],   4),
            })
    return pd.DataFrame(rows).sort_values(["Feature Set", "F1-Score"], ascending=[True, False])


def plot_results_heatmap(results_df: pd.DataFrame, metric="F1-Score"):
    pivot = results_df.pivot(index="Classifier", columns="Feature Set", values=metric)
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 2), 4))
    import seaborn as sns
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0,
    )
    ax.set_title(f"{metric} by Classifier × Feature Set", fontsize=13)
    path = OUTPUT_DIR / f"heatmap_{metric.lower().replace('-','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap saved: %s", path)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import get_X_y
    from feature_engineering import build_tfidf, build_word2vec, build_vader, build_nrc
    from sklearn.model_selection import train_test_split

    X, y, _ = get_X_y()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── feature sets ──────────────────────────────────────────────────────
    tfidf_tr, tfidf_te, _ = build_tfidf(X_tr, X_te)
    w2v_tr,   w2v_te,   _ = build_word2vec(X_tr, X_te)
    vader_tr = build_vader(X_tr); vader_te = build_vader(X_te)
    nrc_tr   = build_nrc(X_tr);   nrc_te   = build_nrc(X_te)
    combined_tr = np.hstack([w2v_tr, vader_tr, nrc_tr])
    combined_te = np.hstack([w2v_te, vader_te, nrc_te])

    feature_sets = {
        "TF-IDF":   (tfidf_tr,    tfidf_te),
        "Word2Vec+Sentiment": (combined_tr, combined_te),
    }

    classifiers = get_classifiers()
    all_results  = {}
    kfold_all    = {}

    for feat_name, (Xf_tr, Xf_te) in feature_sets.items():
        all_results[feat_name] = {}
        for clf_name, clf in classifiers.items():
            # hold-out
            m = train_and_evaluate(
                clf, Xf_tr, y_tr, Xf_te, y_te,
                clf_name=clf_name, feature_name=feat_name,
            )
            all_results[feat_name][clf_name] = m

            # k-fold (on training portion only for speed)
            kfold_key = f"{clf_name} | {feat_name}"
            kfold_all[kfold_key] = stratified_kfold_cv(
                clf, Xf_tr, y_tr, clf_name=kfold_key
            )

    # results table
    df_res = build_results_table(all_results)
    print("\n=== Hold-out Evaluation Results ===")
    print(df_res.to_string(index=False))
    df_res.to_csv(OUTPUT_DIR / "results_table.csv", index=False)

    plot_results_heatmap(df_res)

    # paired t-tests between LR and RF on TF-IDF F1
    print("\n=== Paired T-Tests ===")
    ttest_df = run_all_paired_ttests(kfold_all, metric="f1")
    print(ttest_df[["model_a", "model_b", "mean_a", "mean_b",
                    "t_stat", "p_value", "significant"]].to_string(index=False))
    ttest_df.to_csv(OUTPUT_DIR / "paired_ttests.csv", index=False)
