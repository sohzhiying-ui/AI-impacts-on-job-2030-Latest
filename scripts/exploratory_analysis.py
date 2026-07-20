import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path="../data/AI_Impact_on_Jobs_2030_engineered.csv"):
    df = pd.read_csv(path)
    return df

def summary_statistics(df):
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe(include="all"))

def plot_risk_distribution(df):
    if "Automation_Risk" in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df["Automation_Risk"], bins=20, kde=True)
        plt.title("Distribution of Automation Risk")
        plt.xlabel("Automation Risk")
        plt.ylabel("Frequency")
        plt.show()

def plot_wage_vs_risk(df):
    if {"Wage", "Automation_Risk"} <= set(df.columns):
        plt.figure(figsize=(8,5))
        sns.scatterplot(x="Automation_Risk", y="Wage", data=df, alpha=0.6)
        plt.title("Wage vs Automation Risk")
        plt.xlabel("Automation Risk")
        plt.ylabel("Wage")
        plt.show()

def plot_growth_by_category(df):
    if {"Risk_Category", "Growth_Rate"} <= set(df.columns):
        plt.figure(figsize=(8,5))
        sns.boxplot(x="Risk_Category", y="Growth_Rate", data=df)
        plt.title("Growth Rate by Risk Category")
        plt.xlabel("Risk Category")
        plt.ylabel("Growth Rate")
        plt.show()

if __name__ == "__main__":
    data = load_data()
    summary_statistics(data)
    plot_risk_distribution(data)
    plot_wage_vs_risk(data)
    plot_growth_by_category(data)
