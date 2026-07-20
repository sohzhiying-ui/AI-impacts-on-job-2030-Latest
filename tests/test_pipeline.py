import pandas as pd
from scripts import data_preprocessing, feature_engineering

def test_data_loading():
    df = data_preprocessing.load_data("../data/AI_Impact_on_Jobs_2030.csv")
    assert not df.empty, "Dataset should not be empty"

def test_cleaning():
    df = pd.DataFrame({"Wage": [1000, None, 2000]})
    cleaned = data_preprocessing.clean_data(df)
    assert cleaned["Wage"].isnull().sum() == 0, "Missing wages should be filled"

def test_feature_engineering():
    df = pd.DataFrame({
        "Automation_Risk": [0.2, 0.5, 0.8],
        "Wage": [1000, 2000, 3000],
        "Growth_Rate": [1.5, -0.5, 0.0]
    })
    engineered = feature_engineering.add_risk_category(df)
    assert "Risk_Category" in engineered.columns, "Risk category should be added"
    assert engineered["Risk_Category"].iloc[0] == "Low"
