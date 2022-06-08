#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:49 2022

Random forest baseline with the crisis NLP datasets
RF code from https://github.com/nxs5899/Multi-Class-Text-Classification----Random-Forest/blob/master/multi-class-classifier.ipynb
ROC code used from https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb

@author: amygardiner
"""

#nltk.download('stopwords')
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
  
    
filepath="/Users/amygardiner/Documents/University/PGD/Proj/Data/Paid_labelled/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"

df = pd.read_csv(filepath, sep="\t")
df['tweet_id'] = df['tweet_id'].str.replace("'"," ")

#mapping class strings to integers
numericalmap = {'injured_or_dead_people':1, 'missing_trapped_or_found_people':2, 'displaced_people_and_evacuations':3, 'infrastructure_and_utilities_damage':4, 'donation_needs_or_offers_or_volunteering_services':5, 'caution_and_advice':6, 'sympathy_and_emotional_support':7, 'other_useful_information':8, 'not_related_or_irrelevant':9}
df=df.applymap(lambda s: numericalmap.get(s) if s in numericalmap else s)

train_tweets, test_tweets, y_train, y_test = train_test_split(df['tweet_text'], df['label'], test_size = 0.20, random_state = 0)

c = ['moccasin', 'slategray', 'thistle', 'indianred', 'darkseagreen', 'lightsteelblue', 'khaki', 'powderblue', 'lightpink']
df.groupby('label').tweet_text.count().plot.bar(ylim=0,color=c,title='Distribution of topic classes')
plt.show()

nltk.download('stopwords')
stemmer = PorterStemmer()
words = stopwords.words("english")
df['cleaned'] = df['tweet_text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())


vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(df['cleaned']).toarray()

X = df['cleaned']
Y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', RandomForestClassifier())])

model = pipeline.fit(X_train, y_train)
ytest = np.array(y_test)
y_pred = model.predict(X_test)
y_proba=model.predict_proba(X_test)

classes = model.classes_
roc_auc_ovr = []

for i in range(len(classes)):
    # Gets the class
    c = classes[i]
    df_aux = X_test.copy()
    df_aux['class'] = [1 if y == c else 0 for y in y_test]
    df_aux['prob'] = y_proba[:, i]
    
    try:
        roc_auc_ovr.append(roc_auc_score(df_aux['class'], df_aux['prob']))
    except ValueError:
        pass
    df_aux = df_aux.reset_index(drop = True)
      
print(f'ROC scores: {roc_auc_ovr}')