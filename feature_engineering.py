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

TFIDF_generated = ['bellevu', 'bill', 'boston', 'ca', 'charger', 'cheezburg', 'dc',
                   'diego', 'health', 'houston', 'httpbitlybdkxg', 'httpbitlykxuif',
                   'interior', 'jupdicom', 'lol', 'obama', 'padr', 'redskin', 'redsox',
                   'san', 'sandiego', 'scan', 'sd', 'sdut', 'seattl', 'senat', 'sox',
                   'tcot', 'texa', 'texan', 'twilight', 'tx', 'uw', 'wa', 'washington']

wiki_terms = ['celtics', 'bruins', 'fenway', 'berklee', 'sox', 'astros',
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

###########################################################################################
# TF-IDF
print "TF-IDF processing...\n"
transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(train_tweet_features)
train_features_TFIDF = tfidf.toarray()
print train_features_TFIDF


# np.savetxt('train_tweet_features.txt', train_tweet_features)
# np.savetxt('train_features_TFIDF.txt', train_features_TFIDF)


###########################################################################################
# FEATURE SELECTION


predictors = vocab
X, y = train_features_TFIDF, Preprocess.tweets["city"]
print X.shape

# Perform feature selection
selector = SelectKBest(chi2, k=50)

X_new = selector.fit_transform(X, y)

print X_new.shape
print X_new

# Boolean masking
bool_mask = selector.get_support()
print "\nbool_mask: "
print bool_mask

# Get the selected features
selected_features_chi2 = []
for i in range(len(bool_mask)):
    if bool_mask[i] == True:
        selected_features_chi2.append(vocab[i])
print "\nSelected: "
print selected_features_chi2


'''
#########################################################################################################
print "Training the random forest..."
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)
# Fit the forest to the training set, using the bag of words as features and the city labels as the response variable
forest = forest.fit(train_tweet_features, Preprocess.tweets["city"])

# Read the test data
# Get a bag of words for the test set, and convert to a numpy array
dev_data_features = vectorizer.transform(Preprocess.tweets_dev["cleaned_tweet"])
dev_data_features = dev_data_features.toarray()

# Use the random forest to make predictions
print "Predicting...\n"
result = forest.predict(dev_data_features)

# Copy the results to a pandas dataframe with an "id" column and a "city" column
output = pd.DataFrame(data={"tweet_id": Preprocess.tweets_dev["tweet_id"], "city": result})

# Use pandas to write the comma-separated output file
output.to_csv("Bag_of_Words_model_Random Forest_dev_predict.csv", index=False, quoting=3)

# Evaluate (roughly)
print "Evaluating...\n"

print len(result)
print len(Preprocess.tweets_dev["city"])

same = 0
for i in range(len(result)):
    if result[i] == Preprocess.tweets_dev["city"][i]:
        same += 1
print same
print "Result: " + str(same/len(result))
'''


end = time.time()
print "==============Finished in " + str(end - start) + " seconds.==============\n"
