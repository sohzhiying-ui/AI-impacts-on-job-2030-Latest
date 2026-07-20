import pandas as pd
import joblib
import os

def load_dataset(path="../data/AI_Impact_on_Jobs_2030.csv"):
    """
    Load dataset from CSV file.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Dataset not found at {path}")

def save_dataset(df, path="../data/AI_Impact_on_Jobs_2030_cleaned.csv"):
    """
    Save cleaned or engineered dataset to CSV.
    """
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")

def save_model(model, path="../models/model.pkl"):
    """
    Save trained ML model using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="../models/model.pkl"):
    """
    Load trained ML model.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Model not found at {path}")
