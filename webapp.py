import streamlit as st
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("Sentiment Analysis App")
st.write("Enter a review below to analyze its sentiment (Positive/Negative):")

user_input = st.text_area("Enter your review here:", placeholder="Write your review...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        sentiment = result["label"]
        confidence = round(result["score"], 2)

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence}")
    else:
        st.warning("Please enter a review to analyze.")
