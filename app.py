import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import pickle
import re
import nltk
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Clean text function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text if len(text) > 100 else None
    except:
        return None

def analyze_article(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]
    return prediction, confidence

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .fake-box {
        background-color: #ff4b4b22;
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
    }
    .real-box {
        background-color: #00c85322;
        border: 2px solid #00c853;
        color: #00c853;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Machine Learning & NLP — 99.71% Accuracy</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Paste Article", "🔗 Analyze URL", "📊 Model Performance"])

# ─── Tab 1: Paste Article ───
with tab1:
    article = st.text_area("Enter News Article Here", height=250, placeholder="Paste your news article here...")
    if st.button("🔍 Analyze Article", key="btn1"):
        if article.strip() == "":
            st.warning("Please enter a news article!")
        else:
            with st.spinner("Analyzing..."):
                prediction, confidence = analyze_article(article)
                real_conf = confidence[1] * 100
                fake_conf = confidence[0] * 100

                if prediction == 1:
                    st.markdown('<div class="result-box real-box">✅ REAL NEWS</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-box fake-box">🚨 FAKE NEWS</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Real Confidence", f"{real_conf:.2f}%")
                with col2:
                    st.metric("🚨 Fake Confidence", f"{fake_conf:.2f}%")

                # Confidence chart
                fig, ax = plt.subplots(figsize=(5, 3))
                bars = ax.barh(['Fake', 'Real'], [fake_conf, real_conf],
                               color=['#ff4b4b', '#00c853'])
                ax.set_xlim(0, 100)
                ax.set_xlabel('Confidence %')
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                for bar, val in zip(bars, [fake_conf, real_conf]):
                    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', va='center', color='white')
                st.pyplot(fig)

# ─── Tab 2: URL Analysis ───
with tab2:
    url = st.text_input("🔗 Enter News Article URL", placeholder="https://example.com/news-article")
    if st.button("🔍 Analyze URL", key="btn2"):
        if url.strip() == "":
            st.warning("Please enter a URL!")
        else:
            with st.spinner("Fetching article from URL..."):
                text = extract_text_from_url(url)
                if text is None:
                    st.error("❌ Could not extract article from this URL. Try pasting the text directly.")
                else:
                    st.success(f"✅ Extracted {len(text.split())} words from article")
                    st.text_area("Extracted Text Preview", text[:500] + "...", height=150)

                    prediction, confidence = analyze_article(text)
                    real_conf = confidence[1] * 100
                    fake_conf = confidence[0] * 100

                    if prediction == 1:
                        st.markdown('<div class="result-box real-box">✅ REAL NEWS</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-box fake-box">🚨 FAKE NEWS</div>', unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✅ Real Confidence", f"{real_conf:.2f}%")
                    with col2:
                        st.metric("🚨 Fake Confidence", f"{fake_conf:.2f}%")

# ─── Tab 3: Model Performance ───
with tab3:
    st.markdown("### 🏆 Model Accuracy Comparison")

    models_names = ['Logistic Regression', 'Passive Aggressive', 'Random Forest']
    accuracies = [98.99, 99.64, 99.71]
    colors = ['#0072ff', '#00c6ff', '#00c853']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models_names, accuracies, color=colors, width=0.4)
    ax.set_ylim(98, 100)
    ax.set_ylabel('Accuracy %', color='white')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val}%', ha='center', color='white', fontweight='bold')
    st.pyplot(fig)

    st.markdown("### 📋 Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("📰 Total Articles", "44,898")
    col2.metric("🚨 Fake Articles", "23,481")
    col3.metric("✅ Real Articles", "21,417")