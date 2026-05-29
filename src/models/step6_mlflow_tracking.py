import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import pickle

df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\features.csv")
df["helpful_vote"] = df["helpful_vote"].fillna(0)
df["target"] = (df["rating"] >= 4).astype(int)

features = ["product_idx", "review_length", "verified_flag", "interaction_score", "helpful_vote"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("SmartCart-AI-Experiments")

# Run 1: Random Forest
with mlflow.start_run(run_name="RandomForest_baseline"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.sklearn.log_model(rf, "random_forest_model")
    print(f"RF ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

# Run 2: XGBoost
with mlflow.start_run(run_name="XGBoost_advanced"):
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)[:, 1]

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.xgboost.log_model(xgb, "xgboost_model")
    print(f"XGB ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

print("MLflow tracking complete.")