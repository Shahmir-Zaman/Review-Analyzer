# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 17:23:37 2025

@author: sawee
"""

import pandas as pd  # Essential for loading and exploring your CSV/Excel reviews
import numpy as np   # Useful for handling numerical sentiment scores
import os            # Required to tell Python where your JSON Key is located
# Regex for removing URLs, HTML tags, and special characters
import re            
# # For basic sentiment scoring:
from google.cloud import language_v1 

# Google Gen AI
from google import genai
#Ploting
import matplotlib.pyplot as plt
import seaborn as sns

#credentials for google APi
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nlp-project-482813-4956def5a5f5.json"

# Common Cleaning functions
def text_clean(df,cols):
    #drops NaN vals
    df.dropna(subset=cols, inplace=True)
    
    for col in cols:
        # Convert to lowercase
        df[col] = df[col].astype(str).str.lower()
        #Remove URLs (http, https, www)
        df[col] = df[col].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
        # LOGIC: Remove extra whitespace/newlines
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

# #For Hotel Scinario 
# #EXploring data
# df_hotel= pd.read_csv('hotel_data/Datafiniti_Hotel_Reviews_Jun19.csv')
# # df_hotel.head()
# #print(df_hotel.info())
# #Finding the hotel with most review and storing it
# review_hotel= df_hotel[['name','reviews.rating','reviews.text','reviews.title']]
# review_count = review_hotel['name'].value_counts()
# hotel = review_hotel[review_hotel['name']=='Hyatt House Seattle/Downtown'].copy()
# #Cleaning the stored data 
# text_colums_hotel= ['name','reviews.text','reviews.title']
# hotel = text_clean(hotel,text_colums_hotel)

#Pre-Cleaned Hotel Data
hotel = pd.read_csv('hotel_data/cleaned_hotel_data.csv')




client = language_v1.LanguageServiceClient()

def get_sentiment_score(text):
    # Prepare the document for the API
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    
    # Call the API to analyze sentiment
    annotations = client.analyze_sentiment(request={'document': document})
    score = annotations.document_sentiment.score  # -1.0 to 1.0
    magnitude = annotations.document_sentiment.magnitude  # 0 to infinity (intensity)
    
    return score, magnitude

# Applying it to your Hyatt Hotel dataframe
hotel['sentiment_score'], hotel['intensity'] = zip(*hotel['reviews.text'].apply(get_sentiment_score))



# Initialize Gen AI Client
genai_client = genai.Client(vertexai=True, project="nlp-project-482813", location="us-central1")

def generate_business_insights(positive_reviews, negative_reviews):
    """
    Sends samples of positive and negative reviews to Gemini
    to generate a structured JSON report for a UI dashboard.
    """
    # Take a sample of up to 10 reviews from each category to fit in context
    pos_sample = "\n- ".join(positive_reviews[:10])
    neg_sample = "\n- ".join(negative_reviews[:10])
    
    prompt = f"""
    You are an expert business consultant analyzing customer feedback.
    
    Here is a sample of POSITIVE reviews:
    - {pos_sample}
    
    Here is a sample of NEGATIVE reviews:
    - {neg_sample}
    
    Please analyze these reviews and provide a structured JSON output.
    The response MUST be valid JSON with the following structure:
    {{
      "overall_sentiment_summary": "A concise 1-2 sentence summary of the general sentiment.",
      "key_strengths": [
        "Strength 1 (e.g., Great Location)",
        "Strength 2 (e.g., Friendly Staff)"
      ],
      "areas_for_improvement": [
        "Pain Point 1 (e.g., Noisy Rooms)",
        "Pain Point 2 (e.g., Slow Wi-Fi)"
      ]
    }}
    
    Do not include any markdown formatting (like ```json). Return ONLY the raw JSON string.
    """
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# 1. Classify reviews based on Sentiment Score
# Thresholds: > 0.2 is Positive, < -0.2 is Negative
positive_reviews = hotel[hotel['sentiment_score'] > 0.2]['reviews.text'].tolist()
negative_reviews = hotel[hotel['sentiment_score'] < -0.2]['reviews.text'].tolist()

print(f"DEBUG: Found {len(positive_reviews)} positive and {len(negative_reviews)} negative reviews.")

# 2. Generate insights if we have data
if positive_reviews or negative_reviews:
    print("--- GENERATING BUSINESS INSIGHTS (JSON) ---")
    json_insights = generate_business_insights(positive_reviews, negative_reviews)
    print(json_insights)
else:
    print("Not enough data to generate insights.")


    
    