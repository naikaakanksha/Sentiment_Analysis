import streamlit as st
import joblib
import re

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Predict sentiment
def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction

# Streamlit interface
st.title(" Kindle Review Sentiment Analysis")
st.write("Enter a review below and the model will predict whether it's **positive** or **negative**.")

user_input = st.text_area("Enter Review:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a review!")
    else:
        result = predict_sentiment(user_input)

        if result == "positive":
            st.success("**Positive Review**")
        else:
            st.error("**Negative Review**")

st.markdown("---")
st.write("Built with ❤️ using Streamlit + Machine Learning")
