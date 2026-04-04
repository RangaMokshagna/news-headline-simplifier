# 📰 News Headline Simplifier

> An NLP-powered Streamlit application that analyzes and simplifies complex news headlines using Word-Level, Syntax, and Semantic/Discourse analysis techniques.

---

## 🎯 Project Overview

This project demonstrates a complete NLP pipeline applied to news headlines. It covers three core linguistic analysis layers and combines them to produce simplified, readable versions of complex journalistic language.

**Built for:** NLP course project · Resume · GitHub Portfolio

---

## 🚀 Live Demo

```
streamlit run app.py
```

---

## 🧠 NLP Techniques Used

| Layer | Techniques |
|---|---|
| **Word Level** | Tokenization, POS Tagging, Lemmatization, Stemming, Word Frequency, Syllable Counting |
| **Syntax** | Regex-based Phrase Chunking (NP/VP/PP), Sentence Type Classification, Dependency Roles, Flesch-Kincaid Readability |
| **Semantic & Discourse** | Rule-based NER, Topic Detection, Lexicon Sentiment Analysis, Discourse Marker Detection, Information Density |
| **Search** | TF-IDF Cosine Similarity (from scratch), Keyword Boolean OR/AND |
| **Simplification** | Verbose phrase removal, Complex word substitution, Modifier reduction |

---

## 📁 Project Structure

```
news_headline_simplifier/
│
├── app.py                        ← Main Streamlit entry point
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_dataset.csv        ← 50 complex headlines (publish_date, headline_category, headline_text)
│
├── nlp/
│   ├── __init__.py
│   ├── word_analysis.py          ← Tokenization, POS, Lemma, Stem, Frequency
│   ├── syntax_analysis.py        ← Chunking, Readability, Dependency roles
│   ├── semantic_analysis.py      ← NER, Topics, Sentiment, Discourse
│   └── simplifier.py             ← Core simplification pipeline
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py           ← CSV loading, cleaning, stats
│   └── search.py                 ← TF-IDF + Keyword search (built from scratch)
│
└── components/
    ├── __init__.py
    ├── dashboard.py              ← Dataset overview UI
    ├── analyzer.py               ← Per-headline NLP analysis UI
    ├── search_page.py            ← Search engine UI
    └── batch_page.py             ← Batch simplification UI
```

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/news-headline-simplifier.git
cd news-headline-simplifier

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📂 Dataset Format

Your CSV must contain exactly these three columns:

| Column | Type | Example |
|---|---|---|
| `publish_date` | Date (YYYY-MM-DD) | 2024-01-15 |
| `headline_category` | String | politics |
| `headline_text` | String | Government Officials Deliberate Legislative Amendments... |

A sample dataset with 50 pre-loaded complex headlines is included in `data/sample_dataset.csv`.

---

## ✨ Features

- **Dataset Overview** — Upload or load sample data; view category distributions and word count histograms
- **NLP Analyzer** — Deep analysis of any headline across all three NLP layers, plus simplified output
- **Search Engine** — TF-IDF and Keyword search with category filtering and relevance scoring
- **Batch Simplifier** — Process many headlines at once; download results as CSV

---

## 🛠️ Tech Stack

- Python 3.9+
- [NLTK](https://www.nltk.org/) — NLP core (tokenization, POS, lemmatization, chunking)
- [Streamlit](https://streamlit.io/) — Web UI
- [Pandas](https://pandas.pydata.org/) — Data handling
- TF-IDF implemented from scratch (no sklearn dependency)

---

## 📚 Academic Context

This project covers:
- **Word Level Analysis** — morphological and lexical features
- **Syntax Analysis** — phrase structure and grammatical form
- **Semantic and Discourse Analysis** — meaning, topics, sentiment, coherence
- **Search Techniques** — vector space model (TF-IDF) and Boolean retrieval

---

## 👤 Author

Your Name · [GitHub](https://github.com/RangaMokshagna) 
