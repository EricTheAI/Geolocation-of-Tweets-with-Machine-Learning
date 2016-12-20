import pandas as pd
import re
from nltk.corpus import stopwords
import os
import time
from nltk.stem.snowball import SnowballStemmer
import sys

sys.stdout = open('python_process.log', 'a')

start = time.time()

with open(os.path.join(os.path.dirname(__file__)) + '/data/tweets/train-tweets.txt', 'r') as f:
    # Get rid of "\n" at the end of each line
    data = [line.rstrip() for line in f]

print "THERE ARE " + str(len(data)) + " TWEETS IN TOTAL. \n"

# Store data into DataFrame
uID = []
tID = []
tTEXT = []
tLabel = []
for line in data:
    # Formatting and case folding
    line_spilt = line.lower().split('\t')    # line_spilt = line.lower().split('\t')
    # Ignore lines of tweets which are wrongly formatted.
    if len(line_spilt) == 4:
        uID.append(line_spilt[0])
        tID.append(line_spilt[1])
        tTEXT.append(line_spilt[2])
        tLabel.append(line_spilt[3])

tweet_dict = {
    "user_id": uID,
    "tweet_id": tID,
    "tweet_text": tTEXT,
    "city": tLabel,
}
tweets = pd.DataFrame(tweet_dict)
print "THERE ARE " + str(len(tweets)) + " TWEETS FORMATTED WELL. \n"


# Split text into tokens
def split_text(text):
    """A helping method to split text into tokens."""
    return text.split(' ')
tweets["tokens"] = tweets["tweet_text"].apply(split_text)


# Inspect
# print tweets["tokens"][5]


# Pre-processing Methods

def remove_punctuation(li):
    """Remove punctuation in words."""
    for i in li:
        for j in range(len(i)):
            stripped = re.sub(r'([^\s\w])+', '', i[j])
            i[j] = stripped
    print 'Punctuation removed. \n'


def remove_non_alphabet(li):
    """Remove non-alphabet characters and  in the tokens."""
    total = 0
    pattern = re.compile(r'^([a-zA-Z\s])+$')
    for i in range(len(li)):
        old_len = len(li[i])
        li[i] = filter(lambda x: pattern.match(x), li[i])
        total += (old_len - len(li[i]))
    print 'Removed ' + str(total) + ' non-alphabet characters. \n'


def remove_stop_words(li):
    """Remove stop words."""
    total = 0
    for i in range(len(li)):
        len_old = len(li[i])
        # i = [word for word in i if word not in stopwords.words('english')]
        li[i] = filter(lambda x: x not in stopwords.words('english'), li[i])
        total += (len_old - len(li[i]))
    print 'Removed ' + str(total) + ' stop words. \n'


def stemming(li):
    """Remove morphological affixes from words, leaving only the word stem."""
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for i in li:
        for j in range(len(i)):
            i[j] = str(stemmer.stem(i[j]))
    print 'Words stemmed.'


remove_punctuation(tweets["tokens"])
remove_non_alphabet(tweets["tokens"])
remove_stop_words(tweets["tokens"])
stemming(tweets["tokens"])
remove_stop_words(tweets["tokens"])


# Inspect
# print tweets["tokens"][5]

# Join tokens
def join_tokens(li):
    """A helping method to join tokens."""
    return ' '.join(li)
tweets["cleaned_tweet"] = tweets["tokens"].apply(join_tokens)


# # # Save to .txt
# indexed_tweets = tweets.set_index("tweet_id")
#
# indexed_tweets.to_csv(r'preprocessed.txt', sep='\t')

'''
##############################################################################################################################
# Pre-process test data
with open(os.path.join(os.path.dirname(__file__)) + '/dev-tweets_small.txt', 'r') as f:
    # Get rid of "\n" at the end of each line
    data = [line.rstrip() for line in f]

print "THERE ARE " + str(len(data)) + " dev TWEETS IN TOTAL. \n"

# Store data into DataFrame
uID = []
tID = []
tTEXT = []
tLabel = []
for line in data:
    # Formatting and case folding
    line_spilt = line.lower().split('\t')    # line_spilt = line.lower().split('\t')
    # Ignore lines of tweets which are wrongly formatted.
    if len(line_spilt) == 4:
        uID.append(line_spilt[0])
        tID.append(line_spilt[1])
        tTEXT.append(line_spilt[2])
        tLabel.append(line_spilt[3])

tweet_dev_dict = {
    "user_id": uID,
    "tweet_id": tID,
    "tweet_text": tTEXT,
    "city": tLabel,
}
tweets_dev = pd.DataFrame(tweet_dev_dict)
print "THERE ARE " + str(len(tweets)) + " TWEETS FORMATTED WELL. \n"


# Split text into tokens
def split_text(text):
    """A helping method to split text into tokens."""
    return text.split(' ')
tweets_dev["tokens"] = tweets_dev["tweet_text"].apply(split_text)


remove_punctuation(tweets_dev["tokens"])
remove_non_alphabet(tweets_dev["tokens"])
remove_stop_words(tweets_dev["tokens"])
stemming(tweets_dev["tokens"])
remove_stop_words(tweets_dev["tokens"])


# Join tokens
def join_tokens(li):
    """A helping method to join tokens."""
    return ' '.join(li)
tweets_dev["cleaned_tweet"] = tweets_dev["tokens"].apply(join_tokens)


end = time.time()
print "==============Finished in " + str(end - start) + " seconds.==============\n"

'''
