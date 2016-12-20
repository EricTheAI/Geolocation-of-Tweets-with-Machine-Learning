from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import Preprocess
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys

sys.stdout = open('python_process.log', 'a')

start = time.time()

TFIDF_generated = ['bellevu', 'boston', 'ca', 'charger', 'cheezburg', 'dc',
                   'diego', 'health', 'houston', 'httpbitlybdkxg', 'httpbitlykxuif',
                   'interior', 'jupdicom', 'lol', 'obama', 'padr', 'redskin', 'redsox',
                   'san', 'sandiego', 'scan', 'sd', 'sdut', 'seattl', 'senat', 'sox',
                   'tcot', 'texa', 'texan', 'twilight', 'tx', 'uw', 'wa', 'washington']

wiki_terms = ['celtics', 'fenway', 'berklee', 'sox', 'astros',
            'texans', 'oilers', 'galveston', 'tx', 'rockets', 'los angeles',
            'san francisco','sacramento', 'california', 'san jose', 'supersonics',
            'sounders', 'seahawks', 'intelligencer', 'mariners', 'dc', 'bremerton',
            'redskins', 'yakima']

print "Feature selecting..."
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print "Combining features... \n"

TFIDF_generated.extend(wiki_terms)

print "Removing duplication... \n "
# Remove duplication
features_for_chi2 = list(set(TFIDF_generated))
print "Features for selecting: " + str(len(features_for_chi2)) + "\n"
print features_for_chi2


print "Creating the bag of words...\n"

# Initialize the "CountVectorizer" object.
vectorizer = CountVectorizer(vocabulary=features_for_chi2)

train_tweet_features = vectorizer.fit_transform(Preprocess.tweets["cleaned_tweet"])

# Numpy arrays are easy to work with, so convert the result to an array

train_tweet_features = train_tweet_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

# This is a new feature
.
