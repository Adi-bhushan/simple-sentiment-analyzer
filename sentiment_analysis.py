import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pandas as pd # Kept for future expandability, though not needed for VADER core

# --- Configuration (Minimal) ---
# We will initialize the analyzer inside the prediction function below
# to safely handle NLTK downloads.

# --- Utility Functions ---

# Global variable to store the initialized analyzer, avoiding repeated initialization
GLOBAL_ANALYZER = None

def initialize_vader():
    """Initializes VADER analyzer and downloads lexicon if necessary."""
    global GLOBAL_ANALYZER
    if GLOBAL_ANALYZER is None:
        try:
            # Check if VADER lexicon is downloaded
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            print("[nltk_data] Downloading VADER lexicon...")
            # Streamlit will handle the write permissions for this download
            nltk.download('vader_lexicon', quiet=True) 
        
        GLOBAL_ANALYZER = SentimentIntensityAnalyzer()
        print("VADER Analyzer initialized.")
    return GLOBAL_ANALYZER


def preprocess_text(text):
    """Simple cleaning: VADER is designed to handle common text without extensive pre-processing."""
    # VADER is good at handling social media text, so we only remove residual HTML.
    text = re.sub(r'<.*?>', '', text) 
    return text

def predict_sentiment(text):
    """Preprocesses text, runs VADER prediction, and returns result."""
    
    # Initialize/get analyzer instance (safe method)
    analyzer = initialize_vader()
    
    clean_text = preprocess_text(text)
    vs = analyzer.polarity_scores(clean_text)
    
    # VADER returns 'compound' score from -1 (Extremely Negative) to +1 (Extremely Positive)
    
    # --- Classification Logic (Standard VADER Thresholds) ---
    compound_score = vs['compound']
    
    if compound_score >= 0.05:
        sentiment = "Positive 😊"
        confidence = vs['pos']
    elif compound_score <= -0.05:
        sentiment = "Negative 😞"
        confidence = vs['neg']
    else:
        sentiment = "Neutral 😐"
        confidence = vs['neu']
        
    # Return the predicted category and its respective score
    return sentiment, confidence

if __name__ == '__main__':
    # Test VADER directly when running the script
    initialize_vader()
    
    test_review_1 = "This product is absolutely amazing and exceeded all my expectations. Highly recommend!"
    sentiment_1, confidence_1 = predict_sentiment(test_review_1)
    print(f"\nTest Review: '{test_review_1}'")
    print(f"Predicted Sentiment: {sentiment_1} (Confidence: {confidence_1:.2f})")
    
    test_review_2 = "The delivery was late and the item arrived damaged. Waste of time."
    sentiment_2, confidence_2 = predict_sentiment(test_review_2)
    print(f"\nTest Review: '{test_review_2}'")
    print(f"Predicted Sentiment: {sentiment_2} (Confidence: {confidence_2:.2f})")



