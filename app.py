import streamlit as st
from sentiment_analysis import predict_sentiment 
import time

# --- Streamlit Layout & Styling ---
st.set_page_config(
    page_title="Simple Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Main body background color is white */
    .reportview-container { background: #f0f2f6; } 
    .main { padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    h1 { color: #1f77b4; text-align: center; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 24px; font-size: 16px; border: none; box-shadow: 0 2px #3e8e41; transition: all 0.2s; }
    .stButton>button:hover { background-color: #45a049; }
    .result-box { 
        border: 2px solid #ccc; 
        border-radius: 8px; 
        padding: 15px; 
        margin-top: 20px; 
        /* ADDED: Set the overall text color of the box to black */
        color: black !important;
    }
    /* Ensure paragraph and strong tags are black inside the box */
    .result-box p, .result-box strong {
        color: black !important; 
    }
</style>
""", unsafe_allow_html=True)


def display_result(sentiment, confidence):
    """Displays the analysis result with custom styling."""
    
    # Set color for the border and header based on sentiment
    if "Negative" in sentiment:
        color = "red"
    else:
        color = "green"

    st.markdown(f"""
    <div class="result-box" style="border-color: {color};">
        <h3 style="color: {color}; margin-top: 0;">Sentiment Analysis Result</h3>
        <p><strong>Predicted Category:</strong> <span style="font-size: 1.2em;">{sentiment}</span></p>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)


# --- Streamlit App Layout ---
st.title("Simple NLP Sentiment Analyzer")
st.markdown("Enter a product review below to determine its sentiment (Positive/Neutral or Negative).")

# Text Area for Input
user_input = st.text_area(
    "Paste Review Text Here:",
    placeholder="e.g., The customer service was exceptionally helpful and the product arrived on time.",
    height=150
)

# Analyze Button
if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Show a loading state
        with st.spinner('Analyzing text...'):
            time.sleep(0.5) 
            
            # 2. Get the prediction
            sentiment, confidence = predict_sentiment(user_input)
            
        # 3. Display the result
        display_result(sentiment, confidence)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("Project built using **TF-IDF Vectorization** and **Logistic Regression (Baseline Model)**.")