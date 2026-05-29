from datasets import load_dataset
import pandas as pd
import os

# ── Load dataset ──────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_All_Beauty",
    trust_remote_code=True
)

df = ds["full"].to_pandas()
print(f"Shape: {df.shape}")
print(df.head(3))

# ── Save as CSV ───────────────────────────────────────────────
os.makedirs(r"C:\Users\Shreeji\SmartCart-AI\data", exist_ok=True)
df.to_csv(r"C:\Users\Shreeji\SmartCart-AI\data\amazon_beauty.csv", index=False)
print("Saved to data/amazon_beauty.csv")

# ── Basic EDA ─────────────────────────────────────────────────
print("\n── Column Info ──")
print(df.dtypes)

print("\n── Null Values ──")
print(df.isnull().sum())

print("\n── Rating Distribution ──")
print(df["rating"].value_counts().sort_index())

print("\n── Top 10 Products by Review Count ──")
print(df["asin"].value_counts().head(10))

print("\n── Unique Users:", df["user_id"].nunique())
print("── Unique Products:", df["asin"].nunique())