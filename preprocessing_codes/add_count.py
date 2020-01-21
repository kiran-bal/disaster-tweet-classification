import re
import pandas as pd


#import the test data
test_df = pd.read_csv("../dataset/test.csv")
print(test_df)

test = test_df[['text']]
test_save = test_df[['text']]

#getting only one text
test = test.iloc[30:50,]

#functions to get the count of hashtags,mentions
def hash_count(tweet):
    w = tweet.split()
    return len([word for word in w if word.startswith('#')])

def mention_count(tweet):
    w = tweet.split()
    return len([word for word in w if word.startswith('@')])

def avg_word_len(tweet):
    w = tweet.split()
    word_len = [len(word) for word in w]
    return sum(word_len)/len(word_len)

print("\n\ncheckpoint reached")


#calculating the count by splitting the tweets and adding as extra fields

test['no_chars'] = test['text'].apply(len)
test['no_words'] = test['text'].str.split().apply(len)
test['no_sent'] = test['text'].str.split('.').apply(len)
test['no_para'] = test['text'].str.split('\n').apply(len)
test['avg_word_len'] = test['text'].apply(avg_word_len)
test['no_hashtags'] = test['text'].apply(hash_count)
test['no_mentions'] = test['text'].apply(mention_count)

print("\nafter calculating counts\n")
print(test)
test.to_csv("./temp_output/count_added.csv",index=False)
