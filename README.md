# Fake News Detection System

A modular, end-to-end pipeline for fake news classification using the
**FakeNewsNet** dataset (PolitiFact + GossipCop).

---

## Project Structure

```
fake_news_detection/
├── data_loader.py          # Load & clean CSVs → unified DataFrame
├── eda.py                  # Exploratory Data Analysis (plots + stats)
├── feature_engineering.py  # TF-IDF · Word2Vec · BERT · VADER · NRC
├── models.py               # LR · RF · SVM · k-fold · paired t-test
├── visualize.py            # ROC · PR curves · bar charts · heatmaps
├── main.py                 # Orchestrates the full pipeline
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> BERT support requires `sentence-transformers` and `torch`.
> Skip those if you only want TF-IDF / Word2Vec.

### 2. Place dataset files

Put the four CSVs in the same directory as the scripts **or** set
`FAKENEWS_DATA_DIR`:

```
politifact_fake.csv
politifact_real.csv
gossipcop_fake.csv
gossipcop_real.csv
```

Expected columns (auto-detected):
`id`, `title` / `text` / `content`, and optionally `news_url`.

---

## Running

### Full pipeline (recommended)

```bash
python main.py
```

### With BERT embeddings (slow without GPU)

```bash
python main.py --bert
```

### Single source only

```bash
python main.py --sources politifact
python main.py --sources gossipcop
```

### Skip EDA

```bash
python main.py --no-eda
```

### Custom data directory

```bash
FAKENEWS_DATA_DIR=/path/to/data python main.py
```

---

## Individual Modules

Each file can also be run standalone:

```bash
python data_loader.py         # sanity-check CSV loading
python eda.py                 # EDA only (saves to eda_outputs/)
python feature_engineering.py # test feature building
python models.py              # train+eval with TF-IDF + Word2Vec
python visualize.py           # regenerate plots from saved CSVs
```

---

## Outputs

| Directory         | Contents                                              |
|-------------------|-------------------------------------------------------|
| `eda_outputs/`    | Label distribution, word clouds, n-gram charts, etc. |
| `model_outputs/`  | Confusion matrices, ROC/PR curves, heatmaps           |
| `model_outputs/results_table.csv`  | Hold-out metrics for all combos    |
| `model_outputs/kfold_results.csv`  | Per-fold CV summary                |
| `model_outputs/paired_ttests.csv`  | All pairwise t-test results        |

---

## Feature Sets

| Name                 | Dimensionality    | Description                         |
|----------------------|-------------------|-------------------------------------|
| TF-IDF               | ≤ 20,000          | Unigram + bigram TF-IDF             |
| Word2Vec             | 200               | Averaged GloVe/W2V token embeddings |
| TF-IDF+VADER+NRC     | 20,014            | TF-IDF concatenated with sentiment  |
| Word2Vec+VADER+NRC   | 214               | Dense embedding + sentiment         |
| BERT                 | 768               | DistilBERT [CLS] / mean-pool        |
| BERT+VADER+NRC       | 782               | BERT + sentiment features           |

---

## Classifiers

| Classifier          | Key hyperparameters                         |
|---------------------|---------------------------------------------|
| Logistic Regression | C=1.0, solver=saga, max_iter=1000           |
| Random Forest       | 200 trees, unlimited depth                  |
| SVM (LinearSVC)     | C=1.0, Platt scaling for probabilities      |

---

## Evaluation

- **Hold-out** (80/20 stratified split)
- **Metrics**: accuracy, precision, recall, F1-score, ROC-AUC
- **Stratified k-fold** (default k=5)
- **Paired t-tests** (two-sided, α=0.05) on F1 scores across all model pairs

---

## NRC Emotion Lexicon

The system uses the **NRCLex** Python package, which bundles the lexicon.

Alternatively, download the official NRC Word-Emotion Association Lexicon
(`NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`) from:
<https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>

Then pass its path:

```python
from feature_engineering import build_nrc
feats = build_nrc(texts, nrc_path="/path/to/NRC-Emotion-Lexicon.txt")
```

---

## Extending the Pipeline

- **New classifier**: add it to `get_classifiers()` in `models.py`.
- **New feature set**: build an array and add it to `feature_sets` in `main.py`.
- **New metric**: extend the `evaluate()` function in `models.py`.
