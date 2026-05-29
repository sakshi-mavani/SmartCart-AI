import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="SmartCart AI", layout="wide")
st.title("SmartCart AI — Recommendation & Analytics Dashboard")

df = pd.read_csv(r"C:\Users\Shreeji\SmartCart-AI\data\features.csv")

# Sidebar
st.sidebar.header("Filters")
edu_filter = st.sidebar.multiselect(
    "User Education",
    options=df["user_education"].unique(),
    default=df["user_education"].unique()
)
filtered_df = df[df["user_education"].isin(edu_filter)]

# KPI metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(filtered_df):,}")
col2.metric("Unique Users", f"{filtered_df['user_id'].nunique():,}")
col3.metric("Unique Products", f"{filtered_df['asin'].nunique():,}")
col4.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")

st.markdown("---")

# Rating distribution
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Rating Distribution")
    rating_counts = filtered_df["rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    fig1 = px.bar(rating_counts, x="Rating", y="Count", color="Count", color_continuous_scale="blues")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Reviews by Education Segment")
    edu_counts = filtered_df["user_education"].value_counts().reset_index()
    edu_counts.columns = ["Education", "Count"]
    fig2 = px.pie(edu_counts, names="Education", values="Count", color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Sentiment distribution
col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Sentiment Distribution")
    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    filtered_df["sentiment_label"] = filtered_df["sentiment"].map(sentiment_map)
    sent_counts = filtered_df["sentiment_label"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    fig3 = px.bar(sent_counts, x="Sentiment", y="Count", color="Sentiment",
                  color_discrete_map={"Positive": "#2196F3", "Neutral": "#90CAF9", "Negative": "#1565C0"})
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Top 10 Products by Reviews")
    top_products = filtered_df["asin"].value_counts().head(10).reset_index()
    top_products.columns = ["Product ASIN", "Review Count"]
    fig4 = px.bar(top_products, x="Review Count", y="Product ASIN",
                  orientation="h", color="Review Count", color_continuous_scale="blues")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Live recommendation via API
st.subheader("Live Product Recommendation")
asin_input = st.text_input("Enter Product ASIN", value="0005946468")
top_n = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/recommend",
            json={"asin": asin_input, "top_n": top_n}
        )
        result = response.json()
        if result["cold_start"]:
            st.warning("Cold start: product not found, showing popular items.")
        st.success("Recommended Products:")
        for i, rec in enumerate(result["recommendations"], 1):
            st.write(f"{i}. {rec}")
    except Exception as e:
        st.error(f"API error: {e}")