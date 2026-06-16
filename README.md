# 🍔 McDonald's Customer Reviews — Sentiment Analysis (NLP)

An NLP pipeline that classifies customer reviews as **positive, negative or neutral** — and
goes a step further to surface the recurring *themes* behind each sentiment, the part that
actually tells a business what to fix.

> **Business question:** *What are customers happy and unhappy about — and can we classify
> and explain that sentiment automatically, at scale?*

<!-- TODO: export a chart (e.g. model-accuracy comparison or sentiment distribution) to
     reports/figures/ and link it here: -->
![Model accuracy comparison](reports/figures/01_model_comparison.png)

---

## 🔑 Key Results

| Area | Insight |
|---|---|
| 🏆 Best models | **Logistic Regression** & **Passive Aggressive** — **~91–92% accuracy** |
| 🧪 Models compared | Logistic Regression, Naïve Bayes, Passive Aggressive Classifier |
| 👎 Negative drivers | **Service, order mistakes, waiting time** |
| 👍 Positive drivers | Food quality, clean environment, friendly staff |
| 📏 Evaluation | Accuracy, precision, recall and F1 — not accuracy alone |

---

## 🗂️ Workflow

| Stage | What it does |
|---|---|
| Data preprocessing | Tokenisation, stopword removal, **TF-IDF vectorisation** |
| Sentiment classification | Train & compare three ML classifiers on the labelled reviews |
| Model evaluation | Accuracy, precision, recall, F1 across classes |
| Insights & visualisation | Most common words per sentiment; drivers of positive vs negative reviews |

---

## 📊 Visual Highlights

<!-- Export these from the notebook into reports/figures/, then keep the lines below.
     If you don't want images yet, delete this section. -->

**Sentiment distribution across the review set.**

![Sentiment distribution](reports/figures/02_sentiment_distribution.png)

**Top terms driving negative sentiment — service and order accuracy dominate.**

![Negative sentiment terms](reports/figures/03_negative_terms.png)

---

## 🧰 Tech Stack

`Python` · `pandas` · `numpy` · `NLTK` · `scikit-learn` · `Matplotlib` · `Seaborn`

---

## ▶️ How to Run

```bash
# 1. Clone
git clone https://github.com/shashank-s-k/Customer-Sentiment-Analysis.git
cd Customer-Sentiment-Analysis

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis (notebook or script)
jupyter lab            # open and run the notebook, OR:
# python sentiment_analysis.py
```

---

## 📁 Repository Structure

```
Customer-Sentiment-Analysis/
├── sentiment_analysis.ipynb   # preprocessing → TF-IDF → models → evaluation → insights
├── data/                      # McDonald's reviews dataset
├── reports/
│   └── figures/               # exported PNG charts used in this README
├── requirements.txt
└── README.md
```
*Adjust filenames to match your repo.*

---

## ⚖️ Methodology Notes (the honest bits)

- **TF-IDF + linear classifiers as strong, interpretable baselines:** fast to train and easy
  to explain *why* a review was classified a certain way — valuable when the goal is insight,
  not just a score.
- **Accuracy isn't the whole story:** with uneven class sizes, per-class precision/recall and
  F1 matter more than headline accuracy, so all are reported.
- **Insight over score:** the project deliberately extracts the *drivers* of each sentiment
  (service, wait times, food quality) rather than stopping at a classification metric.

## 🔭 Future Improvements
- Aspect-based sentiment analysis (rate *service* vs *food* vs *price* separately)
- n-gram and embedding-based features
- Deep-learning models (e.g. fine-tuned transformers) for a fair accuracy comparison
