import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, os

df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\features.csv")
df["text"] = df["text"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)
df["combined_text"] = df["title"] + " " + df["text"]

# Build product profiles
product_profiles = df.groupby("asin")["combined_text"].apply(
    lambda x: " ".join(x.values[:10])
).reset_index()
product_profiles.columns = ["asin", "profile"]

# Use top 3000 products only to avoid memory overflow
product_profiles = product_profiles.head(3000).reset_index(drop=True)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(product_profiles["profile"])

# Compute cosine similarity on reduced matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function with cold-start fallback
def get_recommendations(asin, top_n=5):
    if asin not in product_profiles["asin"].values:
        popular = df.groupby("asin")["rating"].mean().nlargest(top_n)
        return popular.index.tolist()
    idx = product_profiles[product_profiles["asin"] == asin].index[0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [product_profiles.iloc[i[0]]["asin"] for i in scores]

# Sample output
sample_asin = product_profiles["asin"].iloc[0]
recs = get_recommendations(sample_asin)
print(f"Input product  : {sample_asin}")
print(f"Recommendations: {recs}")

# Evaluate Precision@5
precision_scores = []
for asin in product_profiles["asin"].sample(200, random_state=42):
    recs = get_recommendations(asin, top_n=5)
    actual_rating = df[df["asin"] == asin]["rating"].mean()
    rec_ratings = df[df["asin"].isin(recs)]["rating"].mean()
    precision_scores.append(1 if rec_ratings >= actual_rating else 0)

print(f"Precision@5: {np.mean(precision_scores):.4f}")

# Save models
os.makedirs(r"C:\Users\Shreeji\SmartCart-AI\models", exist_ok=True)

with open(r"C:\Users\Shreeji\SmartCart-AI\models\tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open(r"C:\Users\Shreeji\SmartCart-AI\models\cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

product_profiles.to_csv(r"C:\Users\Shreeji\SmartCart-AI\data\product_profiles.csv", index=False)
print("Models saved.")