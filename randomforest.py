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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score #inkeeping with literature

class LemmaTokenizer(object):
    
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        
    def __call__(self, tweets):
        return [self.wnl.lemmatize(self.wnl.lemmatize(self.wnl.lemmatize(tok, pos='n'), pos='v'), pos='a') for tok in word_tokenize(tweets)]
    
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())


filepath="/Users/amygardiner/Documents/University/PGD/Proj/Data/Paid_labelled/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"

df = pd.read_csv(filepath, sep="\t")
df['tweet_id'] = df['tweet_id'].str.replace("'"," ")

example_sent=df['tweet_text'][0]

sent_processed=re.sub("@[A-Za-z0-9_]+","",example_sent) #regular expression to remove mentions
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

print(example_sent)       
print(word_tokens)
print(filtered_sentence)


#vectorizer.fit(train_tweets)
#X_train = vectorizer.transform(train_tweets)
#X_test = vectorizer.transform(test_tweets)

"""

x = df.drop(['label'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('ROC Score:', roc_auc_score(y, y_pred))
"""