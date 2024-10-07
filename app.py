import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('C:/Users/HP/Desktop/Practice/spam_comment_classifier.pkl')

# Load the TF-IDF vectorizer (make sure to save this during training and load here)
tfidf_vectorizer = joblib.load('C:/Users/HP/Desktop/Practice/tfidf_vectorizer.pkl')

st.title('YouTube Spam Comment Detector')

# Get user input
user_input = st.text_area('Enter the YouTube comment text')

if st.button('Predict'):
    # Transform user input using the TF-IDF vectorizer
    user_input_transformed = tfidf_vectorizer.transform([user_input])

    # Make prediction
    prediction = model.predict(user_input_transformed)

    # Display result
    if prediction[0] == 1:
        st.write('This comment is classified as: **Spam**')
    else:
        st.write('This comment is classified as: **Not Spam**')
