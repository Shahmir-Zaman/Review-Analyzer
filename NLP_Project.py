# -*- coding: utf-8 -*-
"""
NLP Review Analyzer - Simplified Version
A beginner-friendly dashboard for analyzing customer reviews.
"""

import streamlit as st
import pandas as pd
import os
import re
import time
import json
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import language_v1
from google import genai

# ============================================================
# STEP 1: PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="NLP Review Analyzer",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for right panel styling
st.markdown("""
<style>
    /* Right analytics panel styling */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stMetric"]) {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 3px solid #3b82f6;
        border-radius: 12px;
        padding: 1rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STEP 2: CONNECT TO GOOGLE CLOUD
# ============================================================
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nlp-project-482813-4956def5a5f5.json"

try:
    lang_client = language_v1.LanguageServiceClient()
    genai_client = genai.Client(vertexai=True, project="nlp-project-482813", location="us-central1")
except Exception as e:
    st.error(f"Failed to connect to Google Cloud: {e}")
    lang_client = None
    genai_client = None

# ============================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean a single text string."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def get_sentiment(text):
    """Get sentiment score for a text using Google Cloud NLP."""
    try:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        result = lang_client.analyze_sentiment(request={'document': document})
        return result.document_sentiment.score
    except:
        return 0.0


def get_top_keywords(reviews_list, top_n=10):
    """Extract top keywords from a list of reviews. Returns list of (word, score)."""
    if not reviews_list or len(reviews_list) < 3:
        return []
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
        tfidf_matrix = vectorizer.fit_transform(reviews_list)
        
        scores = tfidf_matrix.sum(axis=0).tolist()[0]
        words = vectorizer.get_feature_names_out()
        
        keyword_scores = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_n]
    except:
        return []


def get_health_status(positive_pct, negative_pct):
    """Determine business health based on sentiment percentages."""
    if negative_pct >= 40:
        return "🔴 Critical - Needs Immediate Attention"
    elif negative_pct >= 25:
        return "🟠 At Risk - Improvement Needed"
    elif positive_pct >= 60:
        return "🟢 Healthy - Strong Performance"
    else:
        return "🟡 Moderate - Room for Growth"


def find_reviews_with_keyword(df, review_col, keyword, limit=3):
    """Find reviews containing a specific keyword."""
    matches = df[df[review_col].str.contains(keyword, case=False, na=False)]
    return matches[review_col].head(limit).tolist()


def generate_ai_insights(pos_keywords, neg_keywords):
    """Generate AI insights using Gemini."""
    prompt = f'''
    Analyze these customer review keywords and provide insights.
    
    POSITIVE KEYWORDS: {pos_keywords}
    NEGATIVE KEYWORDS: {neg_keywords}
    
    Return a JSON with this exact format:
    {{
      "summary": "One sentence summary",
      "strengths": ["strength 1", "strength 2", "strength 3"],
      "improvements": ["improvement 1", "improvement 2", "improvement 3"]
    }}
    
    Return ONLY the JSON, no markdown.
    '''
    
    # Try up to 3 times with delays
    for attempt in range(3):
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash-exp",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return None
    return None


# ============================================================
# STEP 4: SIDEBAR (Controls)
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620986.png", width=60)
    st.title("🧠 NLP Analyzer")
    st.markdown("---")
    
    # Dataset Selection
    dataset = st.selectbox(
        "📁 Select Dataset",
        ["Hotel Reviews","Resturant Reviews" ,"Game Reviews", "Upload Custom"]
    )
    
    # Load data based on selection
    df = None
    review_col = None
    
    if dataset == "Hotel Reviews":
        if os.path.exists('hotel_data/cleaned_hotel_data.csv'):
            df = pd.read_csv('hotel_data/cleaned_hotel_data.csv')
            review_col = 'reviews.text'
        else:
            st.error("Hotel data file not found!")
            
    elif dataset == "Game Reviews":
        if os.path.exists('game_data/cleaned_dataset.csv'):
            df = pd.read_csv('game_data/cleaned_dataset.csv')
            review_col = 'review_text'
        else:
            st.error("Game data file not found!")

    elif dataset == "Resturant Reviews":
        if os.path.exists('zomato_data\cleaned_restaurant_data.csv'):
            df = pd.read_csv('zomato_data\cleaned_restaurant_data.csv')
            review_col = 'Review'
        else:
            st.error("Game data file not found!")
            
    elif dataset == "Upload Custom":
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            review_col = st.selectbox("Select review column:", df.columns)
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True)

# ============================================================
# STEP 5: MAIN CONTENT AREA
# ============================================================

# Filter data by company if selected
df_filtered = df.copy() if df is not None else None

# Show header
if df_filtered is not None:
    st.title("📊 Business Review Analysis")
    
    st.caption(f"Analyzing {len(df_filtered)} reviews")

# Run analysis when button is clicked
if analyze_btn and df_filtered is not None and review_col is not None:
    
    # Create columns: Main (70%) | Right Panel (30%)
    col_main, col_right = st.columns([7, 3])
    
    with st.spinner("Analyzing reviews..."):
        
        # Clean the text
        df_filtered[review_col] = df_filtered[review_col].apply(clean_text)
        df_filtered = df_filtered.dropna(subset=[review_col])
        
        # Sample if too large
        if len(df_filtered) > 300:
            df_sample = df_filtered.sample(300, random_state=42)
        else:
            df_sample = df_filtered
        
        # Get sentiment scores
        progress = st.progress(0, text="Scoring sentiment...")
        scores = []
        for i, text in enumerate(df_sample[review_col]):
            scores.append(get_sentiment(text))
            progress.progress((i + 1) / len(df_sample))
        
        df_sample['sentiment_score'] = scores
        progress.empty()
        
        # Categorize
        df_sample['category'] = df_sample['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )
        
        # Count categories
        pos_reviews = df_sample[df_sample['category'] == 'Positive']
        neg_reviews = df_sample[df_sample['category'] == 'Negative']
        neu_reviews = df_sample[df_sample['category'] == 'Neutral']
        
        total = len(df_sample)
        pos_pct = (len(pos_reviews) / total * 100) if total > 0 else 0
        neg_pct = (len(neg_reviews) / total * 100) if total > 0 else 0
        neu_pct = (len(neu_reviews) / total * 100) if total > 0 else 0
        
        # Get keywords
        pos_keywords = get_top_keywords(pos_reviews[review_col].tolist())
        neg_keywords = get_top_keywords(neg_reviews[review_col].tolist())
    
    # ========== MAIN COLUMN ==========
    with col_main:
        
        # Negative Keywords Bar Chart
        st.subheader("🔴 Top Negative Keywords")
        if neg_keywords:
            neg_df = pd.DataFrame(neg_keywords, columns=['Keyword', 'Score'])
            fig_neg = px.bar(
                neg_df, x='Score', y='Keyword', orientation='h',
                color_discrete_sequence=['#ef4444']
            )
            fig_neg.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("Not enough negative reviews for analysis.")
        
        # Positive Keywords Bar Chart
        st.subheader("🟢 Top Positive Keywords")
        if pos_keywords:
            pos_df = pd.DataFrame(pos_keywords, columns=['Keyword', 'Score'])
            fig_pos = px.bar(
                pos_df, x='Score', y='Keyword', orientation='h',
                color_discrete_sequence=['#22c55e']
            )
            fig_pos.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("Not enough positive reviews for analysis.")
        
        st.markdown("---")
        
        # Deep Dive Section
        st.subheader("🔍 Deep Dive - Sample Reviews")
        
        dive_col1, dive_col2 = st.columns(2)
        
        with dive_col1:
            st.markdown("#### 🔴 Negative Review Clusters")
            if neg_keywords:
                for keyword, score in neg_keywords[:3]:
                    with st.expander(f"📌 {keyword.title()}"):
                        reviews = find_reviews_with_keyword(df_sample, review_col, keyword)
                        if reviews:
                            for r in reviews:
                                st.caption(f'"{r[:200]}..."')
                        else:
                            st.caption("No direct matches found.")
        
        with dive_col2:
            st.markdown("#### 🟢 Positive Review Clusters")
            if pos_keywords:
                for keyword, score in pos_keywords[:3]:
                    with st.expander(f"📌 {keyword.title()}"):
                        reviews = find_reviews_with_keyword(df_sample, review_col, keyword)
                        if reviews:
                            for r in reviews:
                                st.caption(f'"{r[:200]}..."')
                        else:
                            st.caption("No direct matches found.")
    
    # ========== RIGHT PANEL ==========
    with col_right:
        st.subheader("📈 Analytics")
        
        # Pie Chart
        pie_data = pd.DataFrame({
            'Category': ['Positive', 'Negative', 'Neutral'],
            'Count': [len(pos_reviews), len(neg_reviews), len(neu_reviews)]
        })
        
        fig_pie = px.pie(
            pie_data, values='Count', names='Category',
            color='Category',
            color_discrete_map={
                'Positive': '#22c55e',
                'Negative': '#ef4444',
                'Neutral': '#f59e0b'
            }
        )
        fig_pie.update_layout(height=250, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")

        # Health Status
        st.markdown("### 🏥 Health Status")
        health = get_health_status(pos_pct, neg_pct)
        st.info(health)

        
        # KPI Stats
        st.markdown("### 📊 Statistics")
        st.metric("Total Reviews", total)
        st.metric("Positive", f"{len(pos_reviews)} ({pos_pct:.1f}%)")
        st.metric("Negative", f"{len(neg_reviews)} ({neg_pct:.1f}%)")
        st.metric("Neutral", f"{len(neu_reviews)} ({neu_pct:.1f}%)")
        
        st.markdown("---")
        

# Show instructions if no data
elif df is None:
    st.title("📊 Business Review Analyzer")
    st.info("👈 Select a dataset from the sidebar to begin.")