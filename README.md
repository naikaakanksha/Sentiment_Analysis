📝 NLP Sentiment Analysis
🔍 Overview

The NLP Sentiment Analysis project uses Natural Language Processing (NLP) to classify text into Positive, Negative, or Neutral sentiments.
It processes textual data (e.g., Kindle reviews) and predicts sentiment using machine learning models, with an interactive Streamlit web app for real-time predictions.

🚀 Features

Perform Exploratory Data Analysis (EDA) on textual data.

Text preprocessing: cleaning, tokenization, stopword removal, and lemmatization.

Train and evaluate machine learning models for sentiment classification.

Save trained models and TF-IDF vectorizer for reuse.

Interactive predictions using Streamlit or Flask web app.

🧠 Tech Stack

Language: Python

Libraries: pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn, Streamlit

Deployment: Streamlit web app

🏗️ Project Structure
Sentiment_Analysis/
│
├── 04_Streamlit_App.ipynb       # Streamlit web app notebook
├── all_kindle_review.csv        # Raw dataset
├── app.py                       # Web app script
├── cleaned_reviews.csv          # Preprocessed dataset
├── Data_preprocess.ipynb        # Notebook for text preprocessing
├── model_training.ipynb         # Notebook for model training
├── sentiment_model.pkl          # Saved trained model
├── tex_preprocess.py            # Text preprocessing helper script
└── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer

🌱 Usage
Input text in the app to get sentiment predictions.

📊 Example Prediction
Text	Predicted Sentiment
"The book was fantastic and very engaging!"	Positive
"I did not enjoy the story, very boring."	Negative
"It was an average read, nothing special."	Neutral
🧩 Future Enhancements

Integrate deep learning models like LSTM or BERT for higher accuracy.

Support multilingual sentiment analysis.

Deploy on Streamlit Cloud, Heroku, or AWS for real-time usage.

Connect with Twitter API or other platforms for live sentiment analysis.

🪪 License

This project is open-source under the MIT License.