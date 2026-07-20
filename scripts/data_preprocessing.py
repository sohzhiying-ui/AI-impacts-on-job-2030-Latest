# scripts/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path="../data/AI_Impact_on_Jobs_2030.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Example: fill missing wages with median
    if "Wage" in df.columns:
        df["Wage"].fillna(df["Wage"].median(), inplace=True)
    
    # Example: encode categorical columns
    if "Industry" in df.columns:
        le = LabelEncoder()
        df["Industry"] = le.fit_transform(df["Industry"])
    
    return df

def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

if __name__ == "__main__":
    data = load_data()
    data = clean_data(data)
    data = scale_features(data, ["Wage", "Automation_Risk", "Growth_Rate"])
    print(data.head())
