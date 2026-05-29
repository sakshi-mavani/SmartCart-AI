import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle, os

#Load features
print("Loading features...")
df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\features.csv")

#Target: high rating = 1 (4-5 stars), low = 0 (1-3 stars) 
df["target"] = (df["rating"] >= 4).astype(int)

#Feature matrix 
features = [
    "product_idx", "review_length",
    "verified_flag", "interaction_score", "helpful_vote"
]
df["helpful_vote"] = df["helpful_vote"].fillna(0)
X = df[features]
y = df["target"]

#Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

#Model 1: Logistic Regression 
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

print("\n── Logistic Regression Results ──")
print(classification_report(y_test, lr_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

#Model 2: Random Forest 
print("\nTraining Random Forest")
rf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n── Random Forest Results ──")
print(classification_report(y_test, rf_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

#Feature Importance 
print("\n── Feature Importance (Random Forest) ──")
for feat, imp in sorted(
    zip(features, rf.feature_importances_),
    key=lambda x: x[1], reverse=True
):
    print(f"  {feat}: {imp:.4f}")

#Save models 
os.makedirs(r"C:\Users\Shreeji\SmartCart-AI\models", exist_ok=True)
with open(r"C:\Users\Shreeji\SmartCart-AI\models\lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)
with open(r"C:\Users\Shreeji\SmartCart-AI\models\rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\nModels saved to models/")