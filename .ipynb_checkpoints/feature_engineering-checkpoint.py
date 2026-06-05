"""
feature_engineering.py
=======================
Textual representations and sentiment / emotion features.

Provides:
    build_tfidf(X_train, X_test)
    build_word2vec(X_train, X_test)
    build_bert(X_train, X_test)
    build_vader(texts)
    build_nrc(texts)
    build_all_features(X_train, X_test)  ← combines everything

All build_* functions return numpy arrays (train, test) except sentiment
helpers which take a single array and return a (N, D) array.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════

def build_tfidf(X_train, X_test, max_features=20_000, ngram_range=(1, 2)):
    """
    Fit TF-IDF on X_train, transform both splits.

    Returns
    -------
    (train_array, test_array, vectorizer)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    log.info("Building TF-IDF (max_features=%d, ngram=%s) …", max_features, ngram_range)
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    tr = vec.fit_transform(X_train)
    te = vec.transform(X_test)
    log.info("TF-IDF shapes: train=%s  test=%s", tr.shape, te.shape)
    return tr, te, vec


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Word2Vec (averaged embeddings)
# ═══════════════════════════════════════════════════════════════════════════════

def _avg_w2v(sentences, model, dim):
    """Average Word2Vec embeddings for a list of tokenized sentences."""
    vecs = []
    for tokens in sentences:
        valid = [model.wv[t] for t in tokens if t in model.wv]
        vecs.append(np.mean(valid, axis=0) if valid else np.zeros(dim))
    return np.array(vecs)


def build_word2vec(X_train, X_test, vector_size=200, window=5, min_count=2, workers=4):
    """
    Train a Word2Vec model on X_train, then create document embeddings
    by averaging token vectors.

    Returns
    -------
    (train_array, test_array, model)
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("pip install gensim")

    log.info("Training Word2Vec (dim=%d) …", vector_size)
    train_tok = [t.split() for t in X_train]
    test_tok  = [t.split() for t in X_test]

    model = Word2Vec(
        sentences=train_tok,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10,
    )
    tr = _avg_w2v(train_tok, model, vector_size)
    te = _avg_w2v(test_tok,  model, vector_size)
    log.info("Word2Vec shapes: train=%s  test=%s", tr.shape, te.shape)
    return tr, te, model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BERT (sentence-transformers / HuggingFace)
# ═══════════════════════════════════════════════════════════════════════════════

BERT_MODEL = "distilbert-base-uncased"   # swap for any HF model


def build_bert(X_train, X_test, model_name=BERT_MODEL, batch_size=64, max_length=128):
    """
    Encode texts using a pre-trained transformer via sentence-transformers.
    Falls back to HuggingFace transformers + manual mean-pooling if
    sentence-transformers is not installed.

    Returns
    -------
    (train_array, test_array)
    """
    log.info("Building BERT embeddings (%s) …", model_name)

    # ── try sentence-transformers first (fastest API) ──────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name)
        tr = encoder.encode(list(X_train), batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True)
        te = encoder.encode(list(X_test),  batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True)
        log.info("BERT (sentence-transformers) shapes: train=%s  test=%s", tr.shape, te.shape)
        return tr, te

    except ImportError:
        log.warning("sentence-transformers not found; trying HuggingFace transformers …")

    # ── fallback: HuggingFace transformers + mean-pooling ─────────────────
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model  = AutoModel.from_pretrained(model_name)
        hf_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model.to(device)

        def _encode_batch(texts):
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    out = hf_model(**enc)
                # mean-pool over token dimension
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                all_vecs.append(vecs.cpu().numpy())
            return np.vstack(all_vecs)

        tr = _encode_batch(X_train)
        te = _encode_batch(X_test)
        log.info("BERT (HF) shapes: train=%s  test=%s", tr.shape, te.shape)
        return tr, te

    except ImportError:
        raise ImportError("Install either sentence-transformers or transformers + torch.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VADER Sentiment
# ═══════════════════════════════════════════════════════════════════════════════

def build_vader(texts):
    """
    Compute VADER sentiment scores for each text.

    Returns
    -------
    np.ndarray of shape (N, 4)  [neg, neu, pos, compound]
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError("pip install vaderSentiment")

    log.info("Extracting VADER sentiment features …")
    sia = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        rows.append([s["neg"], s["neu"], s["pos"], s["compound"]])
    arr = np.array(rows, dtype=np.float32)
    log.info("VADER feature shape: %s", arr.shape)
    return arr


VADER_COLS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NRC Emotion Lexicon
# ═══════════════════════════════════════════════════════════════════════════════

NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "negative", "positive", "sadness", "surprise", "trust",
]


def _load_nrc_lexicon(nrc_path=None):
    """
    Load the NRC Emotion Lexicon from a TSV file.

    The official NRC file (NRC-Emotion-Lexicon-Wordlevel-v0.92.txt) has columns:
        word  emotion  association (0 or 1)

    If nrc_path is None we try the NRCLex package as a fallback.

    Returns
    -------
    dict: word -> dict of emotion -> score
    """
    if nrc_path and Path(nrc_path).exists():
        df = pd.read_csv(nrc_path, sep="\t", header=None,
                         names=["word", "emotion", "association"])
        df = df[df["association"] == 1]
        lexicon = {}
        for _, row in df.iterrows():
            lexicon.setdefault(row["word"], {})[row["emotion"]] = 1
        return lexicon

    # ── try NRCLex (pip install nrclex) ────────────────────────────────────
    try:
        from nrclex import NRCLex  # noqa: F401
        return None  # signal: use NRCLex directly
    except ImportError:
        pass

    raise FileNotFoundError(
        "NRC Emotion Lexicon not found.\n"
        "  Option A: download NRC-Emotion-Lexicon-Wordlevel-v0.92.txt and pass its path.\n"
        "  Option B: pip install nrclex"
    )


def build_nrc(texts, nrc_path=None):
    """
    Compute NRC emotion features for each text.

    Returns
    -------
    np.ndarray of shape (N, 10)
    """
    log.info("Extracting NRC emotion features …")
    lexicon = _load_nrc_lexicon(nrc_path)

    if lexicon is None:
        # use NRCLex directly
        from nrclex import NRCLex
        rows = []
        for t in texts:
            obj = NRCLex(str(t))
            freq = obj.affect_frequencies
            rows.append([freq.get(e, 0.0) for e in NRC_EMOTIONS])
    else:
        rows = []
        for t in texts:
            tokens = str(t).lower().split()
            counts = {e: 0 for e in NRC_EMOTIONS}
            for tok in tokens:
                for emo, val in lexicon.get(tok, {}).items():
                    if emo in counts:
                        counts[emo] += val
            total = max(sum(counts.values()), 1)
            rows.append([counts[e] / total for e in NRC_EMOTIONS])

    arr = np.array(rows, dtype=np.float32)
    log.info("NRC feature shape: %s", arr.shape)
    return arr


NRC_COLS = [f"nrc_{e}" for e in NRC_EMOTIONS]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Combined feature builder
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.sparse import issparse, hstack as sp_hstack


def _to_dense(arr):
    return arr.toarray() if issparse(arr) else arr


def build_all_features(X_train, X_test, nrc_path=None, include_bert=False):
    """
    Build and concatenate all feature sets.

    Parameters
    ----------
    X_train, X_test : array-like of str
    nrc_path        : path to NRC lexicon file (optional)
    include_bert    : bool  (slow; requires GPU or patience)

    Returns
    -------
    dict with keys:
        'tfidf'     : (tr_sparse, te_sparse, vectorizer)
        'word2vec'  : (tr_dense, te_dense, w2v_model)
        'bert'      : (tr_dense, te_dense) or None
        'vader_tr'  : (N_train, 4)
        'vader_te'  : (N_test,  4)
        'nrc_tr'    : (N_train, 10)
        'nrc_te'    : (N_test,  10)
        'combined_tr': dense array  [word2vec | vader | nrc]
        'combined_te': dense array
    """
    results = {}

    tfidf_tr, tfidf_te, vec = build_tfidf(X_train, X_test)
    results["tfidf"] = (tfidf_tr, tfidf_te, vec)

    w2v_tr, w2v_te, w2v_model = build_word2vec(X_train, X_test)
    results["word2vec"] = (w2v_tr, w2v_te, w2v_model)

    vader_tr = build_vader(X_train)
    vader_te = build_vader(X_test)
    results["vader_tr"] = vader_tr
    results["vader_te"] = vader_te

    nrc_tr = build_nrc(X_train, nrc_path)
    nrc_te = build_nrc(X_test,  nrc_path)
    results["nrc_tr"] = nrc_tr
    results["nrc_te"] = nrc_te

    if include_bert:
        bert_tr, bert_te = build_bert(X_train, X_test)
        results["bert"] = (bert_tr, bert_te)
    else:
        results["bert"] = None

    # Combined dense feature matrix: Word2Vec + VADER + NRC
    combined_tr = np.hstack([w2v_tr, vader_tr, nrc_tr])
    combined_te = np.hstack([w2v_te, vader_te, nrc_te])
    results["combined_tr"] = combined_tr
    results["combined_te"] = combined_te

    log.info("Combined dense feature shapes: train=%s  test=%s",
             combined_tr.shape, combined_te.shape)
    return results


if __name__ == "__main__":
    from data_loader import get_X_y
    from sklearn.model_selection import train_test_split

    X, y, _ = get_X_y()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    feats = build_all_features(X_train, X_test, include_bert=False)
    print("TF-IDF train shape:", feats["tfidf"][0].shape)
    print("Combined train shape:", feats["combined_tr"].shape)
