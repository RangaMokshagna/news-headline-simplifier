# 📰 News Headline Simplifier

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green)
![NLP](https://img.shields.io/badge/NLP-Project-orange)
![NewsAPI](https://img.shields.io/badge/NewsAPI-Live%20News-purple)

> An NLP-powered Streamlit web application that analyzes and simplifies complex news headlines using Word-Level, Syntax, and Semantic/Discourse analysis — with live news fetching via NewsAPI.

---

## 🎯 Project Overview

This project demonstrates a complete NLP pipeline applied to news headlines. It covers three core linguistic analysis layers and combines them to produce simplified, readable versions of complex journalistic language. It also integrates a live news API to fetch real-time headlines.

**Built for:** NLP Course Project · Resume · GitHub Portfolio

---

## 🚀 Live Demo

```bash
streamlit run app.py
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Dataset Overview** | Upload CSV or load sample data, view category charts and word distributions |
| 🔬 **NLP Analyzer** | Deep 3-layer analysis of any headline with simplified output |
| 🔍 **Search Engine** | TF-IDF and Boolean keyword search with category filtering |
| ⚡ **Batch Simplifier** | Simplify many headlines at once, download results as CSV |
| 🌐 **Live News API** | Fetch real-time headlines from NewsAPI and simplify them instantly |

---

## 🧠 NLP Techniques Used

| Layer | Techniques |
|---|---|
| **Word Level** | Tokenization, POS Tagging, Lemmatization, Stemming, Word Frequency, Syllable Counting |
| **Syntax** | Regex Phrase Chunking (NP/VP/PP), Sentence Type Classification, Dependency Roles, Flesch-Kincaid Readability |
| **Semantic & Discourse** | Rule-based NER, Topic Detection, Lexicon Sentiment Analysis, Discourse Marker Detection, Information Density |
| **Search** | TF-IDF Cosine Similarity (built from scratch), Boolean OR/AND Keyword Search |
| **Simplification** | Verbose phrase removal (57 rules), Noun chain compression (39 rules), Complex word substitution (300+ words), Filler adverb removal |

---

## 📁 Project Structure

```
news_headline_simplifier/
│
├── app.py                        ← Main Streamlit entry point
├── requirements.txt              ← Project dependencies
├── README.md                     ← Project documentation
├── .env                          ← API key (never upload to GitHub)
├── .gitignore
│
├── data/
│   └── sample_dataset.csv        ← 50 complex headlines dataset
│
├── nlp/
│   ├── __init__.py
│   ├── word_analysis.py          ← Tokenization, POS, Lemma, Stem, Frequency
│   ├── syntax_analysis.py        ← Chunking, Readability, Dependency roles
│   ├── semantic_analysis.py      ← NER, Topics, Sentiment, Discourse
│   └── simplifier.py             ← Core simplification pipeline (4 passes)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py           ← CSV loading, cleaning, dataset stats
│   ├── search.py                 ← TF-IDF + Keyword search (built from scratch)
│   └── news_api.py               ← Live news fetching via NewsAPI
│
└── components/
    ├── __init__.py
    ├── dashboard.py              ← Dataset overview UI
    ├── analyzer.py               ← NLP analysis UI
    ├── search_page.py            ← Search engine UI
    └── batch_page.py             ← Batch simplifier UI
```

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/RangaMokshagna/news-headline-simplifier.git
cd news-headline-simplifier

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🌐 Live News API Setup

This project integrates **NewsAPI** to fetch real-time headlines.

### Get a Free API Key

1. Go to [newsapi.org](https://newsapi.org)
2. Click **Get API Key** and sign up free
3. Copy your API key

### Use in the App

- Run the app
- In the sidebar select **🔴 Live News API**
- Paste your API key in the input box
- Select a category (technology, health, sports, etc.)
- Click **Fetch Live Headlines**
- Headlines load instantly and can be analyzed or simplified

### Save Key Locally (Optional)

Create a `.env` file in the project root:

```
NEWS_API_KEY=your_api_key_here
```

> ⚠️ Never upload your `.env` file to GitHub. It is already listed in `.gitignore`.

---

## 📂 Dataset Format

Your CSV must have exactly these three columns:

| Column | Type | Example |
|---|---|---|
| `publish_date` | Date (YYYY-MM-DD) | 2024-01-15 |
| `headline_category` | String | politics |
| `headline_text` | String | Government Officials Deliberate... |

A sample dataset with 50 pre-loaded complex headlines is included in `data/sample_dataset.csv`.

---

## 🔍 How the Simplifier Works

The simplification engine runs 4 ordered string-level passes:

```
Pass 1 → Verbose phrase removal     ("pertaining to" → "on")
Pass 2 → Noun chain compression     ("Artificial Intelligence Algorithms" → "AI")
Pass 3 → Complex word substitution  ("deliberate" → "discuss")
Pass 4 → Filler adverb removal      ("substantially", "extensively" removed)
```

**Example:**

```
BEFORE: Government Officials Deliberate Legislative Amendments
        Pertaining to Fiscal Expenditure Allocations

AFTER:  Government Officials Discuss Law Changes on Financial Funds
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| NLTK | NLP — tokenization, POS, lemmatization, chunking |
| Streamlit | Web UI |
| Pandas | Data handling |
| Requests | NewsAPI HTTP calls |
| python-dotenv | Secure API key management |
| TF-IDF (custom) | Search engine — no sklearn used |
| re (regex) | All simplification pattern matching |

---

## 📚 Academic Context

This project covers all major NLP analysis levels:

- **Word Level Analysis** — morphological and lexical features
- **Syntax Analysis** — phrase structure and grammatical form
- **Semantic and Discourse Analysis** — meaning, topics, sentiment, coherence
- **Search Techniques** — vector space model (TF-IDF) and Boolean retrieval
- **API Integration** — real-world data fetching from NewsAPI

---

## 🔧 Future Improvements

- Replace NLTK with spaCy for better accuracy
- Add transformer-based simplification (T5 / BART)
- Add BLEU/ROUGE score evaluation
- Support multiple languages
- Add user login and search history

---
