import pandas as pd

def add_risk_category(df):
    """
    Convert automation risk scores into categories.
    Example: 0-0.3 = Low, 0.3-0.6 = Medium, 0.6-1.0 = High
    """
    if "Automation_Risk" in df.columns:
        df["Risk_Category"] = pd.cut(
            df["Automation_Risk"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )
    return df

def add_vulnerability_index(df):
    """
    Create a new feature combining wage and automation risk.
    High risk + low wage = more vulnerable.
    """
    if {"Automation_Risk", "Wage"} <= set(df.columns):
        df["Vulnerability_Index"] = df["Automation_Risk"] * (1 / (df["Wage"] + 1))
    return df

def add_growth_flag(df):
    """
    Flag jobs with negative growth rate as 'Declining'.
    """
    if "Growth_Rate" in df.columns:
        df["Growth_Flag"] = df["Growth_Rate"].apply(
            lambda x: "Declining" if x < 0 else "Growing"
        )
    return df

if __name__ == "__main__":
    # Load cleaned dataset
    data = pd.read_csv("../data/AI_Impact_on_Jobs_2030.csv")
    
    # Apply feature engineering steps
    data = add_risk_category(data)
    data = add_vulnerability_index(data)
    data = add_growth_flag(data)
    
    # Preview engineered features
    print(data.head())
    
    # Save engineered dataset
    data.to_csv("../data/AI_Impact_on_Jobs_2030_engineered.csv", index=False)
