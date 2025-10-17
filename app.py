import streamlit as st
from sentiment_analysis import predict_sentiment 
import time

# --- Streamlit Layout & Styling ---
st.set_page_config(
    page_title="VADER Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    h1 { color: #8A2BE2; text-align: center; } /* Purple heading for VADER version */
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 24px; font-size: 16px; border: none; box-shadow: 0 2px #3e8e41; transition: all 0.2s; }
    .result-box { 
        border: 2px solid #ccc; 
        border-radius: 8px; 
        padding: 15px; 
        margin-top: 20px; 
        color: black !important;
    }
    .result-box p, .result-box strong { color: black !important; }
</style>
""", unsafe_allow_html=True)


def display_result(sentiment, confidence):
    """Displays the analysis result with custom styling."""
    
    # Set color based on VADER's 3-class output
    if "Negative" in sentiment:
        color = "red"
    elif "Positive" in sentiment:
        color = "green"
    else:
        color = "darkorange" # Neutral

    st.markdown(f"""
    <div class="result-box" style="border-color: {color};">
        <h3 style="color: {color}; margin-top: 0;">Sentiment Analysis Result</h3>
        <p><strong>Predicted Category:</strong> <span style="font-size: 1.2em;">{sentiment}</span></p>
        <p><strong>Confidence (Score):</strong> {confidence:.2f}</p>
    </div>
    """, unsafe_allow_html=True)


# --- App Layout ---
st.title("Pre-trained VADER Sentiment Analyzer")
st.markdown("This model analyzes text based on a pre-trained, lexicon-based dictionary.")

# Text Area for Input
user_input = st.text_area(
    "Paste Review Text Here:",
    placeholder="e.g., The customer service was helpful, but the shipping was slow.",
    height=150
)

# Analyze Button
if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing text...'):
            time.sleep(0.5) 
            sentiment, confidence = predict_sentiment(user_input)
            
        display_result(sentiment, confidence)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("Project uses **VADER (NLTK)** for zero-dependency, guaranteed deployment.")