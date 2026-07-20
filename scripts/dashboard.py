import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Title
st.title("AI_Impact on Jobs 2030 Dashboard")

# Load dataset
df = pd.read_csv("data/AI_Impact_on_Jobs_2030.csv")

# --- Sidebar interactive features ---
st.sidebar.header("Filters")

# 1. Search box (filter by industry)
search_industry = st.sidebar.text_input("Search Industry")

# 2. Dropdown filter (select country)
country_filter = st.sidebar.selectbox("Select Country", ["All"] + list(df["Country"].unique()))

# Apply filters
filtered_df = df.copy()
if search_industry:
    filtered_df = filtered_df[filtered_df["Industry"].str.contains(search_industry, case=False, na=False)]
if country_filter != "All":
    filtered_df = filtered_df[filtered_df["Country"] == country_filter]


# --- Visualization 1: Histogram of AI_Replacement_Risk ---
st.subheader("AI_Replacement_Risk Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df["AI_Replacement_Risk"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# --- Visualization 2: Scatterplot Average_Salary_USD vs AI_Replacement_Risk ---
st.subheader("Average_Salary_USD vs AI_Replacement_Risk")
fig, ax = plt.subplots()
sns.scatterplot(x="AI_Replacement_Risk", y="Average_Salary_USD", data=filtered_df, ax=ax)
st.pyplot(fig)

# --- Visualization 3: Education_Level vs AI_Replacement_Risk ---
st.subheader("Education_Level vs AI_Replacement_Risk")
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x="Education_Level", y="AI_Replacement_Risk", data=filtered_df, estimator="mean", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)



# --- Predictive / Analytical Output ---
st.subheader("Predictive Model Output")

# Load your trained model (make sure best_model.pkl exists in repo)
try:
    model = joblib.load("best_model.pkl")

    # User input form
    st.write("Enter job parameters for prediction:")
    wage_input = st.number_input("Average_Salary_USD", min_value=0, value=50000)
    risk_input = st.slider("AI_Replacement_Risk", 0.0, 1.0, 0.5)
    demand_input = st.slider("Future Demand Score", 0.0, 1.0, 0.5)

    if st.button("Predict"):
        # Example: adjust features to match your model training
        features = [[wage_input, risk_input, demand_input]]
        prediction = model.predict(features)
        st.success(f"Predicted Job Category: {prediction[0]}")

except Exception as e:
    st.warning("Model not found or error loading model. Please ensure best_model.pkl is in your repo.")
    st.write(e)

