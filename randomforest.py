#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:49 2022

Random forest baseline with the crisis NLP datasets
Code used from https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb

@author: amygardiner
"""

#import nltk
#nltk.download('stopwords')
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr


def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    
filepath="/Users/amygardiner/Documents/University/PGD/Proj/Data/Paid_labelled/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"

df = pd.read_csv(filepath, sep="\t")
df['tweet_id'] = df['tweet_id'].str.replace("'"," ")

#mapping class strings to integers
numericalmap = {'injured_or_dead_people':1, 'missing_trapped_or_found_people':2, 'displaced_people_and_evacuations':3, 'infrastructure_and_utilities_damage':4, 'donation_needs_or_offers_or_volunteering_services':5, 'caution_and_advice':6, 'sympathy_and_emotional_support':7, 'other_useful_information':8, 'not_related_or_irrelevant':9}
df=df.applymap(lambda s: numericalmap.get(s) if s in numericalmap else s)

train_tweets, test_tweets, y_train, y_test = train_test_split(df['tweet_text'], df['label'], test_size = 0.20, random_state = 0)

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

model_multiclass = RandomForestClassifier(n_estimators = 50, criterion = 'gini')
model_multiclass.fit(X_train, y_train)
y_pred = model_multiclass.predict(X_test)
y_proba = model_multiclass.predict_proba(X_test)

classes = model_multiclass.classes_

# Plots the Probability Distributions and the ROC Curves One vs Rest
plt.figure(figsize = (12, 8))
bins = [i/20 for i in range(20)] + [1]
roc_auc_ovr = {}

for i in range(len(classes)):
    # Gets the class
    c = classes[i]
    
    # Prepares an auxiliar dataframe to help with the plots
    df_aux = X_test.copy()
    df_aux['class'] = [1 if y == c else 0 for y in y_test]
    df_aux['prob'] = y_proba[:, i]
    df_aux = df_aux.reset_index(drop = True)
    
    # Plots the probability distribution for the class and the rest
    ax = plt.subplot(2, 3, i+1)
    sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
    ax.set_title(c)
    ax.legend([f"Class: {c}", "Rest"])
    ax.set_xlabel(f"P(x = {c})")
    
    # Calculates the ROC Coordinates and plots the ROC Curves
    ax_bottom = plt.subplot(2, 3, i+4)
    tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
    plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
    ax_bottom.set_title("ROC Curve OvR")
    
    # Calculates the ROC AUC OvR
    roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
    
plt.tight_layout()







