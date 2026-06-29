import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = "data/heart.csv"
PROCESSED_DIR = "ml/processed"

FEATURE_COLS = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalch','exang','oldpeak','slope','ca','thal'
]

TARGET_COL = "num"

SEX_MAP = {'male':1, 'female':0}
CP_MAP = {
    'typical angina':0,
    'atypical angina':1,
    'non-anginal':2,
    'asymptomatic':3
}
FBS_MAP = {True:1, False:0}
RESTECG_MAP = {'normal':0, 'st-t abnormality':1, 'lv hypertrophy':2}
EXANG_MAP = {True:1, False:0}
SLOPE_MAP = {'upsloping':0, 'flat':1, 'downsloping':2}
THAL_MAP = {'normal':1, 'fixed defect':2, 'reversable defect':3}

def preprocess_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df['sex'] = df['sex'].str.lower().map(SEX_MAP)
    df['cp'] = df['cp'].str.lower().map(CP_MAP)
    df['fbs'] = df['fbs'].map(FBS_MAP)
    df['restecg'] = df['restecg'].str.lower().map(RESTECG_MAP)
    df['exang'] = df['exang'].map(EXANG_MAP)
    df['slope'] = df['slope'].str.lower().map(SLOPE_MAP)
    df['thal'] = df['thal'].str.lower().map(THAL_MAP)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].apply(lambda x: 1 if x > 0 else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, f"{PROCESSED_DIR}/scaler.pkl")

    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train_scaled)
    np.save(f"{PROCESSED_DIR}/X_test.npy", X_test_scaled)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train.values)
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test.values)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_pipeline()
