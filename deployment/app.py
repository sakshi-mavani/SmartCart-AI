from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI(title="SmartCart AI API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "models", "tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "cosine_sim.pkl"), "rb") as f:
    cosine_sim = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

product_profiles = pd.read_csv(os.path.join(BASE_DIR, "data", "product_profiles.csv"))

class RecommendRequest(BaseModel):
    asin: str
    top_n: int = 5

class RatingRequest(BaseModel):
    product_idx: float
    review_length: float
    verified_flag: float
    interaction_score: float
    helpful_vote: float

def get_recommendations(asin: str, top_n: int = 5):
    if asin not in product_profiles["asin"].values:
        return {"cold_start": True, "recommendations": []}
    idx = product_profiles[product_profiles["asin"] == asin].index[0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    recs = [product_profiles.iloc[i[0]]["asin"] for i in scores]
    return {"cold_start": False, "recommendations": recs}

@app.get("/")
def root():
    return {"message": "SmartCart AI API is running"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    return get_recommendations(req.asin, req.top_n)

@app.post("/predict-rating")
def predict_rating(req: RatingRequest):
    features = [[
        req.product_idx,
        req.review_length,
        req.verified_flag,
        req.interaction_score,
        req.helpful_vote
    ]]
    pred = xgb_model.predict(features)[0]
    proba = xgb_model.predict_proba(features)[0][1]
    return {
        "predicted_high_rating": bool(pred),
        "confidence": round(float(proba), 4)
    }