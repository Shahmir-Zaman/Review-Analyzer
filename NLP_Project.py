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

#For Hotel Scinario 
#EXploring data
df_hotel= pd.read_csv('hotel_data/Datafiniti_Hotel_Reviews_Jun19.csv')
# df_hotel.head()
#print(df_hotel.info())
#Finding the hotel with most review and storing it
review_hotel= df_hotel[['name','reviews.rating','reviews.text','reviews.title']]
review_count = review_hotel['name'].value_counts()
hotel = review_hotel[review_hotel['name']=='Hyatt House Seattle/Downtown'].copy()
#Cleaning the stored data 
text_colums_hotel= ['name','reviews.text','reviews.title']
hotel = text_clean(hotel,text_colums_hotel)
hotel.to_csv('cleaned_hotel_data.csv', index=False)


    
    