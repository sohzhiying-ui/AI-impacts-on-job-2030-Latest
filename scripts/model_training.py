import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path="data/AI_Impact_on_Jobs_2030.csv"):
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    X = df[["Years_Experience", "Future_Demand_Score", "Job_Growth_2030", "Automation_Level"]]
    
    # Convert numeric AI_Replacement_Risk into categories
    y = pd.cut(
        df["AI_Replacement_Risk"],
        bins=[-float("inf"), 0.10, 0.40, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    
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
 # Save the trained model for Streamlit dashboard
    joblib.dump(model, "best_model.pkl")
    print(" Model saved as best_model.pkl")
