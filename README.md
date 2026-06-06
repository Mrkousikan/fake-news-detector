# 📰 Fake News Detector

An ML-powered web app that classifies news articles as **Real or Fake** with 99.71% accuracy.

## Features
- Paste article text for instant analysis
- Analyze news from any URL
- Confidence score with visualization
- Comparison of 3 ML models

## Tech Stack
- Python, Scikit-learn, NLTK
- TF-IDF Vectorization
- Random Forest Classifier
- Streamlit UI

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## accuracy
- Logistic Regression: 98.99%
- Passive Aggressive: 99.64%
- Random Forest: 99.71%
