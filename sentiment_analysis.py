import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
import re

# --- Configuration ---
try:
    # Attempt to find stopwords, download if missing
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("[nltk_data] Downloading package stopwords...")
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
MODEL_FILE = 'sentiment_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

# --- Utility Functions ---

def preprocess_text(text):
    """Clean the text: remove non-alphanumeric, lower case, and remove stopwords."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(words)

def train_and_save_model():
    """Trains the Logistic Regression model using expanded synthetic data."""
    print("Starting model training with **EXPANDED** sample data...")
    
    # --- EXPANDED SYNTHETIC DATA (16 samples for better feature learning) ---
    data = {
        'review': [
            # POSITIVE/NEUTRAL SAMPLES (Label 1)
            "This product is absolutely amazing and exceeded all my expectations. Highly recommend!",
            "It's okay. Nothing special, but it works as advertised and I'm generally pleased.",
            "A total game-changer! Fast shipping and great quality.",
            "Unbelievably good value for money. Five stars!",
            "I am thrilled with this device; the battery life is phenomenal.",
            "Highly functional and performs exactly as promised. Zero complaints.",
            "I love this item. It is durable and stylish.",
            "Excellent customer service and a fantastic warranty period.",

            # NEGATIVE SAMPLES (Label 0) - More diversity
            "Worst purchase ever. Broke immediately and the support was terrible and unresponsive.",
            "I'm deeply disappointed with the slow performance and poor packaging. A complete rip-off.",
            "Steer clear of this mess. Complete waste of time and resources and the build quality is shoddy.",
            "The item was defective upon arrival and the company refused to refund me. Horrible experience.",
            "I hate this product. It stopped working after one week. Useless and frustrating.",
            "Completely failed to meet my expectations; the features are buggy and unreliable.",
            "The installation process was painful and the device is noisy.",
            "Avoid buying this. The lack of basic quality control is shocking."
        ],
        'sentiment': [1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0] # 1: Positive/Neutral, 0: Negative
    }
    df = pd.DataFrame(data)

    df['clean_review'] = df['review'].apply(preprocess_text)
    
    X = df['clean_review']
    y = df['sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Engineering (TF-IDF Vectorizer)
    vectorizer = TfidfVectorizer(max_features=100) 
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"TF-IDF Vectorizer saved to {VECTORIZER_FILE}")

    # Model Building (Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    # Evaluation
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set (Expanded Data): {accuracy:.2f}")

    joblib.dump(model, MODEL_FILE)
    print(f"Trained Logistic Regression model saved to {MODEL_FILE}")
    
    return model, vectorizer

# Load model and vectorizer once when imported
try:
    GLOBAL_MODEL = joblib.load(MODEL_FILE)
    GLOBAL_VECTORIZER = joblib.load(VECTORIZER_FILE)
except FileNotFoundError:
    print("Model or Vectorizer files not found. Training model now...")
    GLOBAL_MODEL, GLOBAL_VECTORIZER = train_and_save_model()


def predict_sentiment(text):
    """Predicts sentiment using the loaded simple model, reverting to direct prediction (no custom boundary)."""
    clean_text = preprocess_text(text)
    text_vectorized = GLOBAL_VECTORIZER.transform([clean_text])
    
    # Get the model's raw prediction (0 or 1), which automatically uses the >50% rule
    prediction = GLOBAL_MODEL.predict(text_vectorized)[0]
    
    # Get probabilities for both classes (0 and 1)
    probability = GLOBAL_MODEL.predict_proba(text_vectorized)[0]
    
    # Use the model's direct prediction
    if prediction == 1: 
        sentiment = "Positive/Neutral 😊"
        confidence = probability[1] # Use the confidence for class 1
    else:
        sentiment = "Negative 😞"
        confidence = probability[0] # Use the confidence for class 0
        
    return sentiment, confidence

if __name__ == '__main__':
    train_and_save_model()

    test_review_1 = "This is simply the greatest product I have ever bought, the quality is top-notch!"
    sentiment_1, confidence_1 = predict_sentiment(test_review_1)
    print(f"\nTest Review: '{test_review_1}'")
    print(f"Predicted Sentiment: {sentiment_1} (Confidence: {confidence_1:.2f})")
    
    test_review_2 = "The delivery was late and the item arrived damaged."
    sentiment_2, confidence_2 = predict_sentiment(test_review_2)
    print(f"\nTest Review: '{test_review_2}'")
    print(f"Predicted Sentiment: {sentiment_2} (Confidence: {confidence_2:.2f})")
