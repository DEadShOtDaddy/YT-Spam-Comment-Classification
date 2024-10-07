import streamlit as st
import joblib

# Custom CSS styling for a better look
st.markdown(
    """
    <style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        width: 160px;
        margin-top: 20px;
    }
    .stTextArea>label {
        font-size: 18px;
        color: #4CAF50;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load the trained model and vectorizer
model = joblib.load('C:/Users/HP/Desktop/Practice/spam_comment_classifier.pkl')
tfidf_vectorizer = joblib.load('C:/Users/HP/Desktop/Practice/tfidf_vectorizer.pkl')

# Layout organization using columns
col1, col2 = st.columns([1, 3])

with col1:
    st.image('C:/Users/HP/Desktop/YT-Spam-Comment-Classification/pic.png', width=120)


with col2:
    st.title('🌟 YouTube Spam Comment Detector 🌟')

st.markdown("### Analyze comments quickly and accurately with our AI-powered spam detector.")

# Sidebar for additional settings or information
st.sidebar.header('Settings')
st.sidebar.info('Use this tool to classify comments as spam or not spam based on machine learning.')

# Text input for user to enter a YouTube comment
user_input = st.text_area('Enter the YouTube comment text below:', height=150)

# When the 'Predict' button is clicked, make a prediction
if st.button('Predict'):
    if user_input:
        with st.spinner('Analyzing the comment...'):
            user_input_transformed = tfidf_vectorizer.transform([user_input])
            prediction = model.predict(user_input_transformed)

        # Display the result with a beautiful message
        if prediction[0] == 1:
            st.error('⚠️ This comment is classified as: **Spam**')
        else:
            st.success('✅ This comment is classified as: **Not Spam**')
    else:
        st.warning("Please enter a comment to analyze.")
