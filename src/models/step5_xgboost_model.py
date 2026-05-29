import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle, os

df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\features.csv")

df["helpful_vote"] = df["helpful_vote"].fillna(0)
df["target"] = (df["rating"] >= 4).astype(int)

features = ["product_idx", "review_length", "verified_flag", "interaction_score", "helpful_vote"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

print("\nFeature Importances:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

os.makedirs(r"C:\Users\Shreeji\SmartCart-AI\models", exist_ok=True)
with open(r"C:\Users\Shreeji\SmartCart-AI\models\xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")