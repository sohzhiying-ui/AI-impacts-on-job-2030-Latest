import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_data(path="../data/AI_Impact_on_Jobs_2030_engineered.csv"):
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    X = df[["Automation_Risk", "Wage", "Growth_Rate", "Vulnerability_Index"]]
    y = df["Risk_Category"]
    return X, y

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            try:
                auc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class="ovr")
                print(f"ROC-AUC Score: {auc:.3f}")
            except Exception:
                print("ROC-AUC not available for this model.")

if __name__ == "__main__":
    data = load_data()
    X, y = prepare_features(data)
    evaluate_models(X, y)
