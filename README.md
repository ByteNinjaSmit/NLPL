# NLP Lab (NLPL)

Coursework assignments for Natural Language Processing. Each script is standalone and
runnable on its own.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

NLTK `stopwords` are downloaded automatically on first run of each script.

## Assignments

### Assignment_1.py — Text Preprocessing Pipeline
Cleans a raw news article (metadata/date/time stripping, URL, mention, hashtag and
emoji removal) then runs a spaCy + NLTK pipeline: tokenization, stopword removal,
Snowball stemming, lemmatization, POS tagging, and a final clean corpus.

```bash
python Assignment_1.py
```

### Assignment_2.py — Bag-of-Words, TF-IDF & Plagiarism Detection
Builds BoW and TF-IDF from scratch with NumPy, cross-checks against scikit-learn's
`CountVectorizer` / `TfidfVectorizer`, computes cosine similarity between documents,
flags plagiarism above a threshold, and plots top keywords per document.

```bash
python Assignment_2.py
```

### Assignment_3.py — Word Embeddings & Semantic Product Search
Trains a Word2Vec (skip-gram) model on 50 e-commerce product descriptions, explores
similarity and analogies, compares against pre-trained GloVe
(`glove-wiki-gigaword-100`, downloaded on first run), visualizes embeddings with PCA
and t-SNE, and builds a semantic product recommendation engine (query→product and
product→product).

Generated artifacts (git-ignored, regenerable):
- `ecommerce_word2vec.model`
- `ecommerce_product_vectors.npy`
- `ecommerce_products.csv`

```bash
python Assignment_3.py
```

### Assignment_4.py — Topic Modeling on Support Tickets
Runs LSA/SVD and LDA on customer-support tickets, tunes the topic count via `c_v`
coherence, renders per-topic word clouds, a coherence-vs-topics plot, and a pyLDAvis
interactive view, then assigns each ticket to its dominant topic.

```bash
python Assignment_4.py
```

## Requirements

Python 3.9+ recommended. See `requirements.txt`.
