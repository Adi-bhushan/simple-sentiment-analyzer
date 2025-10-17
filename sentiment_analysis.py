import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# --- Configuration ---
# VADER requires the 'vader_lexicon' resource
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("[nltk_data] Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
    
# Initialize the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# --- Utility Functions ---

def preprocess_text(text):
    """Simple cleaning: VADER is designed to handle common text without extensive pre-processing."""
    text = re.sub(r'<.*?>', '', text) # Remove any residual HTML
    return text

def predict_sentiment(text):
    """Predicts sentiment using the VADER pre-trained lexicon."""
    
    clean_text = preprocess_text(text)
    vs = analyzer.polarity_scores(clean_text)
    
    # VADER returns 'compound' score from -1 (Extremely Negative) to +1 (Extremely Positive)
    compound_score = vs['compound']
    
    # --- Classification Logic (Standard VADER Thresholds) ---
    if compound_score >= 0.05:
        sentiment = "Positive 😊"
        # Confidence is mapped from the positive score
        confidence = vs['pos']
    elif compound_score <= -0.05:
        sentiment = "Negative 😞"
        # Confidence is mapped from the negative score
        confidence = vs['neg']
    else:
        sentiment = "Neutral 😐"
        confidence = vs['neu']
        
    # For a unified return, we normalize confidence to the highest score
    # and adjust the sentiment text for the web app.
    return sentiment, max(vs['pos'], vs['neg'], vs['neu'])

if __name__ == '__main__':
    # VADER does not require training, just testing its logic
    
    test_review_1 = "This product is absolutely amazing and exceeded all my expectations. Highly recommend!"
    sentiment_1, confidence_1 = predict_sentiment(test_review_1)
    print(f"\nTest Review: '{test_review_1}'")
    print(f"Predicted Sentiment: {sentiment_1} (Confidence: {confidence_1:.2f})")
    
    test_review_2 = "The delivery was late and the item arrived damaged. Waste of time."
    sentiment_2, confidence_2 = predict_sentiment(test_review_2)
    print(f"\nTest Review: '{test_review_2}'")
    print(f"Predicted Sentiment: {sentiment_2} (Confidence: {confidence_2:.2f})")
    
    test_review_3 = "The device works fine, but nothing special."
    sentiment_3, confidence_3 = predict_sentiment(test_review_3)
    print(f"\nTest Review: '{test_review_3}'")
    print(f"Predicted Sentiment: {sentiment_3} (Confidence: {confidence_3:.2f})")