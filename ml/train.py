import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

PROCESSED_DIR = "ml/processed"
MODEL_DIR = "ml/models"

def load_data():
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test

def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(eval_metric="logloss")
    xgb.fit(X_train, y_train)

    joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(xgb, f"{MODEL_DIR}/xgboost.pkl")

    print("Models trained and saved.")

if __name__ == "__main__":
    train_models()
