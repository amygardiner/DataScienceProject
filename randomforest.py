#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:49 2022

Random forest baseline with the crisis NLP datasets

@author: amygardiner
"""
#import nltk
#nltk.download('stopwords')
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score #inkeeping with literature

filepath="/Users/amygardiner/Documents/University/PGD/Proj/Data/Paid_labelled/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"

df = pd.read_csv(filepath, sep="\t")
df['tweet_id'] = df['tweet_id'].str.replace("'"," ")

#mapping class strings to integers
numericalmap = {'injured_or_dead_people':1, 'missing_trapped_or_found_people':2, 'displaced_people_and_evacuations':3, 'infrastructure_and_utilities_damage':4, 'donation_needs_or_offers_or_volunteering_services':5, 'caution_and_advice':6, 'sympathy_and_emotional_support':7, 'other_useful_information':8, 'not_related_or_irrelevant':9}
df=df.applymap(lambda s: numericalmap.get(s) if s in numericalmap else s)

train_tweets, test_tweets, train_labels, test_labels = train_test_split(df['tweet_text'], df['label'], test_size = 0.20, random_state = 0)

def preprocessor(text):
    tweets=[]
    for i in text:
        sent_processed=re.sub("@[A-Za-z0-9_]+","",i) #regular expression to remove mentions
        sent_processed=re.sub(r'http\S+', '', sent_processed) #regex to remove URLs
        sent_processed=re.sub('RT', '', sent_processed) #removing RT
        sent_processed=re.sub("[^\w\s]", '', sent_processed) #removing punctuation
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sent_processed)
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = []
      
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        tweets.append(filtered_sentence)
    return tweets

train_tweets=preprocessor(train_tweets)
test_tweets=preprocessor(test_tweets)

count_vec = MultiLabelBinarizer()
mlb = count_vec.fit(train_tweets)
X_train = mlb.transform(train_tweets)
X_test = mlb.transform(test_tweets)


regressor = RandomForestRegressor(random_state = 0)
regressor.fit(X_train, train_labels)
y_pred = regressor.predict(X_test)

print('ROC Score:', roc_auc_score(test_labels, y_pred,multi_class='ovr'))
