💬 Simple NLP Sentiment Analyzer for Product Reviews
This project implements a complete Machine Learning workflow to classify product reviews into Positive/Neutral or Negative categories, showcasing core Natural Language Processing (NLP) and deployment skills.

🛠️ Tech Stack & Methods
Language: Python

Core Libraries: scikit-learn, pandas, nltk, joblib

Feature Engineering: Term Frequency-Inverse Document Frequency (TF-IDF) Vectorization

Model: Logistic Regression (Baseline Classification Model)

Deployment: Streamlit (Web Application)

🚀 How to Run Locally
Prerequisites
Python 3.8+ installed.

pip and venv (Virtual Environment).

Installation
Clone this repository and install the dependencies:

git clone [YOUR_GITHUB_REPO_URL]
cd sentiment-analyzer
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Training & Deployment
Train the Model: Run the core script to train the Logistic Regression model on the synthetic data and save the model files (.joblib):

python sentiment_analysis.py

Launch the App: Start the Streamlit web application:

streamlit run app.py

The application will open in your browser, ready for testing.

📁 Project Files
File

Description

sentiment_analysis.py

Contains the data cleaning, model training, and prediction logic. Saves the model assets.

app.py

The Streamlit script that loads the trained model and provides the user interface.

requirements.txt

Lists all required Python packages for deployment.
