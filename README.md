# Customer Feedback Topic Clustering: Unsupervised NLP Pipeline

An end-to-end unsupervised machine learning and NLP pipeline built in Python to automatically discover, segment, and summarize recurring complaint themes across thousands of raw customer app reviews—without requiring pre-labeled training data or external LLM APIs.

---

## Overview

Product and support teams often struggle to manually read through thousands of unstructured user reviews to prioritize bug fixes and UX improvements. This project transforms raw review text into structured, actionable issue clusters using **TF-IDF vectorization**, **TruncatedSVD** dimensionality reduction, and **K-Means clustering** validated via **Silhouette Score**.

### Key Features
* **Automated Text Preprocessing:** Cleans raw user feedback (lowercasing, URL/punctuation stripping, custom domain stopword filtering) to isolate meaningful signal.
* **TF-IDF + Dimensionality Reduction:** Extracts unigram and bigram features (`ngram_range=(1, 2)`) and projects sparse text matrices into a dense latent space using `TruncatedSVD`.
* **Data-Driven Cluster Selection:** Evaluates candidate cluster counts ($k \in [3, 7]$) using Silhouette Score to select optimal topic separation automatically.
* **Actionable Keyword Extraction & Reporting:** Computes mean TF-IDF weights per cluster to extract top representative keywords, sample reviews, and visual distribution charts.

---

## Project Structure

```text
customer_feedback_clustering/
├── data/
│   └── reviews.csv                  # Input dataset (auto-generated sample data if not provided)
├── outputs/
│   ├── top_keywords_per_cluster.csv # Cluster counts, percentage shares, top terms & sample reviews
│   └── cluster_overview.png         # Bar chart of cluster sizes & Silhouette Score curve
├── cluster_reviews.ipynb            # End-to-end interactive Jupyter Notebook
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Tech Stack

* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **NLP & Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `TruncatedSVD`, `KMeans`, `silhouette_score`)
* **Visualization:** Matplotlib

---

## Quickstart

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/yash-mk-pawar/customer-feedback-clustering.git
cd customer-feedback-clustering
pip install -r requirements.txt
```

### 2. Prepare Data (Optional)
* Place your own CSV file at `data/reviews.csv` containing a `review_text` column (and optional `rating` column).
* If `data/reviews.csv` is not present, running the notebook automatically generates **5,000 realistic synthetic app reviews** across five product complaint themes (*Crash & Performance, Payments & Refunds, Login & Account Access, UI & Navigation, Customer Support & Delivery*).

### 3. Run the Notebook
Open `cluster_reviews.ipynb` in Jupyter Notebook or VS Code and run all cells (`Run All`).

---

## Pipeline Workflow

1. **Ingestion & Cleaning:** Loads `data/reviews.csv`, drops null/short entries, and normalizes text.
2. **Feature Engineering:** Vectorizes cleaned text into a 1,500-feature TF-IDF matrix (1–2 n-grams) and compresses to 30 components via `TruncatedSVD`.
3. **Clustering & Evaluation:** Evaluates $k = 3 \dots 7$ with Silhouette Score, fits the optimal `KMeans` model, and assigns cluster labels to every review.
4. **Export & Visualization:** Saves `outputs/top_keywords_per_cluster.csv` and `outputs/cluster_overview.png` for downstream product reporting.

---

## Outputs Generated

* **`outputs/top_keywords_per_cluster.csv`**: Summary table mapping each discovered cluster ID to review count, percentage share, top 8 defining keywords, and a representative customer quote.
* **`outputs/cluster_overview.png`**: Two-panel diagnostic chart showing review distribution across clusters alongside the Silhouette Score optimization curve.

---

## Future Enhancements

* Add **HDBSCAN** or **UMAP** visualization for non-spherical complaint clusters and outlier/novel issue detection.
* Connect cluster outputs to a lightweight **Streamlit** or **Power BI** dashboard to track complaint volume trends over time and by app version.
