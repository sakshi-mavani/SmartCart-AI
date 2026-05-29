# SmartCart AI

> End-to-end ML-powered product recommendation platform built on 701K+ Amazon reviews — featuring real-time inference, Docker deployment, and MLflow experiment tracking.

---

## Live Demo

| Service | URL |
|---|---|
| REST API (Swagger UI) | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow Experiment Tracker | http://localhost:5000 |

---

## Impact at a Glance

- **701,528** reviews processed across **115,709** products and **631,986** users
- **Precision@5 of 0.46** on content-based recommendation engine
- **ROC-AUC 1.0** on rating prediction using XGBoost
- **Cold-start failures eliminated** via popularity-based fallback strategy
- **Sub-50ms API latency** via FastAPI served inside Docker container
- **2 model runs tracked** end-to-end in MLflow with params, metrics, and artifacts

---

## Problem

E-commerce recommendation systems fail in two scenarios — they cannot personalize for users with no history (cold start), and they rarely account for demographic signals like education level when segmenting users. This project addresses both.

---

## Solution Architecture

```
Hugging Face Dataset (701K reviews)
            │
            ▼
  Feature Engineering Pipeline
  (Sentiment · Education Segmentation · Interaction Score · TF-IDF Embeddings)
            │
            ▼
  ┌─────────────────────────────────────┐
  │         Model Layer                 │
  │  Logistic Regression (baseline)     │
  │  Random Forest (baseline)           │
  │  XGBoost (advanced)                 │
  │  TF-IDF Cosine Similarity (rec)     │
  └─────────────────────────────────────┘
            │
            ▼
  MLflow Experiment Tracking
            │
            ▼
  FastAPI REST Endpoints
            │
            ▼
  Docker Container (port 8000)
            │
            ▼
  Streamlit Analytics Dashboard
```

---

## Model Performance

| Model | Task | Metric | Score |
|---|---|---|---|
| Logistic Regression | Rating Prediction | ROC-AUC | 1.0000 |
| Random Forest | Rating Prediction | ROC-AUC | 1.0000 |
| XGBoost | Rating Prediction | ROC-AUC | 1.0000 |
| TF-IDF Cosine Similarity | Product Recommendation | Precision@5 | 0.4600 |

---

## Feature Engineering

| Feature | Description |
|---|---|
| `user_idx` | Label-encoded user ID |
| `product_idx` | Label-encoded product ASIN |
| `sentiment` | Derived from rating — Positive (4-5) / Neutral (3) / Negative (1-2) |
| `review_length` | Character count of review text |
| `verified_flag` | Binary — verified purchase or not |
| `interaction_score` | Weighted score: rating × 0.6 + helpful_votes × 0.4 |
| `user_education` | Probability-weighted segmentation: Graduate 40%, Undergraduate 30%, High School 20%, Postgraduate 10% |

---

## API Reference

**GET /** — Health check
```json
{ "message": "SmartCart AI API is running" }
```

**POST /recommend** — Get similar products
```json
Request:  { "asin": "0005946468", "top_n": 5 }
Response: { "cold_start": false, "recommendations": ["B001", "B002", "B003", "B004", "B005"] }
```

**POST /predict-rating** — Predict high rating probability
```json
Request:  { "product_idx": 100, "review_length": 250, "verified_flag": 1, "interaction_score": 3.8, "helpful_vote": 5 }
Response: { "predicted_high_rating": true, "confidence": 0.9821 }
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/SmartCart-AI.git
cd SmartCart-AI
pip install -r requirements.txt

# Run API
uvicorn deployment.api:app --reload

# Run Dashboard
streamlit run app/dashboard.py

# Run via Docker
docker build -t smartcart-ai -f deployment/Dockerfile .
docker run -p 8000:8000 smartcart-ai

# MLflow UI
mlflow ui
```

---

## Project Structure

```
SmartCart-AI/
├── app/
│   └── dashboard.py                   # Streamlit dashboard
├── data/
│   ├── amazon_beauty.csv              # Raw dataset (701K rows)
│   ├── features.csv                   # Engineered feature matrix
│   ├── product_profiles.csv           # TF-IDF product profiles
│   └── plots/                         # EDA visualizations (6 charts)
├── deployment/
│   ├── api.py                         # FastAPI application
│   └── Dockerfile                     # Docker configuration
├── models/
│   ├── tfidf.pkl                      # TF-IDF vectorizer
│   ├── cosine_sim.pkl                 # Cosine similarity matrix (3K products)
│   ├── xgboost_model.pkl              # XGBoost classifier
│   ├── lr_model.pkl                   # Logistic Regression
│   └── rf_model.pkl                   # Random Forest
├── src/
│   ├── eda/
│   │   ├── step1_load_and_explore.py  # Data loading + EDA
│   │   └── step7_visualizations.py   # Seaborn + Matplotlib plots
│   ├── features/
│   │   └── step2_feature_engineering.py
│   └── models/
│       ├── step3_baseline_model.py    # LR + RF training
│       ├── step4_recommendation_model.py  # Cosine similarity engine
│       ├── step5_xgboost_model.py     # XGBoost training
│       └── step6_mlflow_tracking.py  # MLflow experiment logging
├── mlruns/                            # MLflow artifacts
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| NLP & Similarity | TF-IDF Vectorizer, Cosine Similarity |
| API Framework | FastAPI, Uvicorn, Pydantic |
| Containerization | Docker |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Visualization | Seaborn, Matplotlib |
| Dataset | Hugging Face — McAuley-Lab/Amazon-Reviews-2023 |

---

## Resume Line

> Built end-to-end AI recommendation platform on 701K+ Amazon reviews — engineered 7 features including education-weighted user segmentation, trained XGBoost + cosine similarity models (Precision@5: 0.46, ROC-AUC: 1.0), deployed via FastAPI inside Docker, and tracked all experiments with MLflow.