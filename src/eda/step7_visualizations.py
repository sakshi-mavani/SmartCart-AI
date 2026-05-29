import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\features.csv")
df["helpful_vote"] = df["helpful_vote"].fillna(0)

os.makedirs(r"C:\Users\Shreeji\SmartCart-AI\data\plots", exist_ok=True)

# Plot 1: Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="rating", data=df, palette="Blues_d")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Shreeji\SmartCart-AI\data\plots\rating_distribution.png")
plt.close()
print("Plot 1 saved.")

# Plot 2: Sentiment Distribution
plt.figure(figsize=(8, 5))
sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
df["sentiment_label"] = df["sentiment"].map(sentiment_map)
sns.countplot(x="sentiment_label", data=df, palette="Blues_d", order=["Positive", "Neutral", "Negative"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Shreeji\SmartCart-AI\data\plots\sentiment_distribution.png")
plt.close()
print("Plot 2 saved.")

# Plot 3: Education Segment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="user_education", data=df, palette="Blues_d",
              order=df["user_education"].value_counts().index)
plt.title("User Education Segment Distribution")
plt.xlabel("Education")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Shreeji\SmartCart-AI\data\plots\education_distribution.png")
plt.close()
print("Plot 3 saved.")

# Plot 4: Top 10 Products by Review Count
plt.figure(figsize=(10, 6))
top10 = df["asin"].value_counts().head(10)
sns.barplot(x=top10.values, y=top10.index, palette="Blues_d")
plt.title("Top 10 Products by Review Count")
plt.xlabel("Review Count")
plt.ylabel("Product ASIN")
plt.tight_layout()
plt.savefig(r"C:\Users\Shreeji\SmartCart-AI\data\plots\top10_products.png")
plt.close()
print("Plot 4 saved.")

# Plot 5: Rating vs Review Length
plt.figure(figsize=(8, 5))
sns.boxplot(x="rating", y="review_length", data=df[df["review_length"] < 2000], palette="Blues")
plt.title("Rating vs Review Length")
plt.xlabel("Rating")
plt.ylabel("Review Length (chars)")
plt.tight_layout()
plt.savefig(r"C:\Users\Shreeji\SmartCart-AI\data\plots\rating_vs_review_length.png")
plt.close()
print("Plot 5 saved.")

# Plot 6: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_cols = ["rating", "review_length", "verified_flag", "interaction_score", "helpful_vote"]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(r"C:\Users\Shreeji\SmartCart-AI\data\plots\correlation_heatmap.png")
plt.close()
print("Plot 6 saved.")

print("All visualizations saved to data/plots/")