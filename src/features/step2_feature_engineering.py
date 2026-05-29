import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

#Load saved CSV 
print("Loading data...")
df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\amazon_beauty.csv")
print(f"Shape: {df.shape}")

#1. Clean nulls
df["text"]  = df["text"].fillna("")
df["title"] = df["title"].fillna("")

#2. Encode user_id and asin to integers 
user_enc    = LabelEncoder()
product_enc = LabelEncoder()
df["user_idx"]    = user_enc.fit_transform(df["user_id"])
df["product_idx"] = product_enc.fit_transform(df["asin"])

#3. Sentiment score from rating 
# 4-5 = positive, 3 = neutral, 1-2 = negative
df["sentiment"] = df["rating"].apply(
    lambda x: 1 if x >= 4 else (0 if x == 3 else -1)
)

#4. Review length feature 
df["review_length"] = df["text"].apply(len)

#5. Verified purchase flag 
df["verified_flag"] = df["verified_purchase"].astype(int)

#6. Education segmentation (your MBA project logic)
np.random.seed(42)
edu_map = {
    "graduate":       0.40,
    "undergraduate":  0.30,
    "high_school":    0.20,
    "postgraduate":   0.10
}
df["user_education"] = np.random.choice(
    list(edu_map.keys()),
    size=len(df),
    p=list(edu_map.values())
)

#7. Interaction score (weighted rating) 
df["interaction_score"] = (
    df["rating"] * 0.6 +
    df["helpful_vote"].fillna(0) * 0.4
)

#Save engineered features 
out_path = r"C:\Users\Shreeji\SmartCart-AI\data\features.csv"
df.to_csv(out_path, index=False)
print(f"Features saved → {out_path}")

#Summary
print("\n── New Columns Added ──")
print(df[["user_idx","product_idx","sentiment",
          "review_length","verified_flag",
          "user_education","interaction_score"]].head(5))
print("\nDone! Shape:", df.shape)