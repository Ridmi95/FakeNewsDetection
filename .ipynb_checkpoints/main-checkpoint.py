"""
main.py
=======
End-to-end Fake News Detection pipeline.

Steps:
    1. Load & preprocess data (data_loader.py)
    2. Exploratory Data Analysis (eda.py)
    3. Feature engineering — TF-IDF, Word2Vec, BERT*, VADER, NRC (feature_engineering.py)
    4. Train Logistic Regression, Random Forest, SVM (models.py)
    5. Evaluate with accuracy / precision / recall / F1 / ROC-AUC
    6. Stratified k-fold cross-validation
    7. Pairwise paired t-tests

* BERT is disabled by default (slow without GPU).  Enable via --bert flag.

Usage examples
--------------
# Quick run (no BERT, both sources)
    python main.py

# Include BERT embeddings
    python main.py --bert

# Only PolitiFact data
    python main.py --sources politifact

# Skip EDA
    python main.py --no-eda

# Custom data directory
    FAKENEWS_DATA_DIR=/path/to/data python main.py
"""

import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse, hstack as sp_hstack

from data_loader import get_X_y
from eda import run_full_eda
from feature_engineering import (
    build_tfidf, build_word2vec, build_bert,
    build_vader, build_nrc,
)
from models import (
    get_classifiers, train_and_evaluate,
    stratified_kfold_cv, run_all_paired_ttests,
    build_results_table, plot_results_heatmap,
    OUTPUT_DIR as MODEL_OUT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_FOLDS      = 5


# ── argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fake News Detection Pipeline")
    p.add_argument(
        "--sources", nargs="+", default=["politifact", "gossipcop"],
        choices=["politifact", "gossipcop"],
        help="Which dataset sources to include.",
    )
    p.add_argument("--bert",   action="store_true", help="Include BERT embeddings.")
    p.add_argument("--no-eda", action="store_true", help="Skip EDA step.")
    p.add_argument("--folds",  type=int, default=N_FOLDS, help="Number of CV folds.")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _dense(arr):
    return arr.toarray() if issparse(arr) else arr


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    log.info("=" * 60)
    log.info("Fake News Detection – full pipeline")
    log.info("Sources : %s", args.sources)
    log.info("BERT    : %s", args.bert)
    log.info("Folds   : %d", args.folds)
    log.info("=" * 60)

    # ── 1. Data loading ───────────────────────────────────────────────────
    X, y, df = get_X_y(args.sources)
    log.info("Dataset shape: %s  |  fake=%d  real=%d",
             df.shape, y.sum(), (y == 0).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # ── 2. EDA ────────────────────────────────────────────────────────────
    if not args.no_eda:
        run_full_eda(df)

    # ── 3. Feature engineering ────────────────────────────────────────────
    log.info("--- Feature Engineering ---")
    tfidf_tr, tfidf_te, tfidf_vec = build_tfidf(X_train, X_test)
    w2v_tr,   w2v_te,   w2v_model = build_word2vec(X_train, X_test)
    vader_tr  = build_vader(X_train);  vader_te  = build_vader(X_test)
    nrc_tr    = build_nrc(X_train);    nrc_te    = build_nrc(X_test)

    # TF-IDF + VADER + NRC (sparse + dense → dense)
    tfidf_vader_nrc_tr = np.hstack([_dense(tfidf_tr), vader_tr, nrc_tr])
    tfidf_vader_nrc_te = np.hstack([_dense(tfidf_te), vader_te, nrc_te])

    # Word2Vec + VADER + NRC
    combined_tr = np.hstack([w2v_tr, vader_tr, nrc_tr])
    combined_te = np.hstack([w2v_te, vader_te, nrc_te])

    feature_sets = {
        "TF-IDF":                (tfidf_tr,         tfidf_te),
        "Word2Vec":              (w2v_tr,            w2v_te),
        "TF-IDF+VADER+NRC":     (tfidf_vader_nrc_tr, tfidf_vader_nrc_te),
        "Word2Vec+VADER+NRC":   (combined_tr,        combined_te),
    }

    if args.bert:
        bert_tr, bert_te = build_bert(X_train, X_test)
        feature_sets["BERT"] = (bert_tr, bert_te)
        bert_comb_tr = np.hstack([bert_tr, vader_tr, nrc_tr])
        bert_comb_te = np.hstack([bert_te, vader_te, nrc_te])
        feature_sets["BERT+VADER+NRC"] = (bert_comb_tr, bert_comb_te)

    # ── 4 & 5. Training + hold-out evaluation ────────────────────────────
    log.info("--- Training & Evaluation ---")
    classifiers  = get_classifiers()
    all_results  = {}  # {feat_name: {clf_name: metrics_dict}}
    kfold_all    = {}  # {key: {metric: fold_scores}}

    for feat_name, (Xf_tr, Xf_te) in feature_sets.items():
        all_results[feat_name] = {}
        for clf_name, clf in classifiers.items():
            m = train_and_evaluate(
                clf, Xf_tr, y_train, Xf_te, y_test,
                clf_name=clf_name, feature_name=feat_name,
            )
            all_results[feat_name][clf_name] = m

    # ── 6. Stratified k-fold CV ───────────────────────────────────────────
    log.info("--- Stratified %d-Fold Cross-Validation ---", args.folds)
    # Run CV on each feature set × classifier
    for feat_name, (Xf_tr, _) in feature_sets.items():
        for clf_name, clf in classifiers.items():
            key = f"{clf_name} | {feat_name}"
            kfold_all[key] = stratified_kfold_cv(
                clf, Xf_tr, y_train,
                n_splits=args.folds,
                clf_name=key,
            )

    # Save k-fold summary
    kfold_rows = []
    for key, scores in kfold_all.items():
        clf_name, feat_name = key.split(" | ", 1)
        for metric, arr in scores.items():
            kfold_rows.append({
                "Classifier": clf_name,
                "Feature Set": feat_name,
                "Metric": metric,
                "Mean": round(arr.mean(), 4),
                "Std":  round(arr.std(),  4),
                "Folds": args.folds,
            })
    kfold_df = pd.DataFrame(kfold_rows)
    kfold_df.to_csv(MODEL_OUT / "kfold_results.csv", index=False)
    log.info("K-fold results saved to model_outputs/kfold_results.csv")

    # ── 7. Paired t-tests ─────────────────────────────────────────────────
    log.info("--- Paired T-Tests ---")
    ttest_df = run_all_paired_ttests(kfold_all, metric="f1")
    ttest_df.to_csv(MODEL_OUT / "paired_ttests.csv", index=False)
    log.info("Paired t-test results saved to model_outputs/paired_ttests.csv")

    # ── Summary outputs ───────────────────────────────────────────────────
    results_df = build_results_table(all_results)
    results_df.to_csv(MODEL_OUT / "results_table.csv", index=False)
    plot_results_heatmap(results_df, metric="F1-Score")
    plot_results_heatmap(results_df, metric="ROC-AUC")

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOLD-OUT EVALUATION RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"STRATIFIED {args.folds}-FOLD CV SUMMARY (F1)")
    print("=" * 70)
    f1_cv = kfold_df[kfold_df["Metric"] == "f1"][
        ["Classifier", "Feature Set", "Mean", "Std"]
    ].sort_values("Mean", ascending=False)
    print(f1_cv.to_string(index=False))

    print("\n" + "=" * 70)
    print("PAIRED T-TESTS (F1, α=0.05)")
    print("=" * 70)
    cols = ["model_a", "model_b", "mean_a", "mean_b", "t_stat", "p_value", "significant"]
    print(ttest_df[cols].to_string(index=False))

    log.info("Pipeline complete.  Outputs in: %s", MODEL_OUT.resolve())


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(parse_args())
