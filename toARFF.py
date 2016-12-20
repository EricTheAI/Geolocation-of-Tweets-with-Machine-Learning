from sklearn.feature_extraction.text import CountVectorizer
import arff
import Preprocess
import sys
import numpy as np

sys.stdout = open('python_process.log', 'a')

chi2_generated = ['httpbitlybdkxg', 'houston', 'san', 'scan', 'boston', 'ca', 'washington', 'httpbitlykxuif',
                   'padr', 'seattl', 'redsox', 'sox', 'texa', 'diego', 'texan', 'bellevu', 'wa', 'tx', 'jupdicom',
                   'lol', 'bremerton', 'health', 'sdut', 'interior', 'senat', 'redskin', 'sounders', 'redskins',
                   'los angeles', 'tcot', 'sacramento', 'astros', 'san jose', 'yakima', 'dc', 'california', 'seahawks',
                   'galveston', 'celtics', 'twilight', 'obama', 'sandiego', 'charger', 'uw', 'bill', 'cheezburg',
                   'texans', 'fenway', 'supersonics', 'sd']

vectorizer = CountVectorizer(vocabulary=chi2_generated)

X = vectorizer.fit_transform(Preprocess.tweets["cleaned_tweet"])

# Numpy arrays are easy to work with, so convert the result to an array

X_array = X.toarray()
X_array_np = np.array(X_array)
X_array_np_list = X_array_np.tolist()
print "Generating ARFF file for Weka ..."
##############################################################################
# add class to the end
for i in range(len(X_array)):
    X_array_np_list[i].append(Preprocess.tweets["city"][i])


######################################################
titles = ['httpbitlybdkxg', 'houston', 'san', 'scan', 'boston', 'ca', 'washington', 'httpbitlykxuif',
            'padr', 'seattl', 'redsox', 'sox', 'texa', 'diego', 'texan', 'bellevu', 'wa', 'tx', 'jupdicom',
            'lol', 'bremerton', 'health', 'sdut', 'interior', 'senat', 'redskin', 'sounders', 'redskins',
            'los angeles', 'tcot', 'sacramento', 'astros', 'san jose', 'yakima', 'dc', 'california', 'seahawks',
            'galveston', 'celtics', 'twilight', 'obama', 'sandiego', 'charger', 'uw', 'bill', 'cheezburg',
            'texans', 'fenway', 'supersonics', 'sd', 'location']

data = X_array_np_list
arff.dump('train_50.arff', data, relation="train_50", names=titles)
