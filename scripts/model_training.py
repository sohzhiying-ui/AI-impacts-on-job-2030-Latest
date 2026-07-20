import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path="../data/AI_Impact_on_Jobs_2030_engineered.csv"):
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    # Example target: Risk_Category (Low, Medium, High)
    X = df[["Automation_Risk", "Wage", "Growth_Rate", "Vulnerability_Index"]]
    y = df["Risk_Category"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    data = load_data()
    X, y = prepare_features(data)
    model = train_model(X, y)
