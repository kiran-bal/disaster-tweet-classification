import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # importing tensorflow library
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os

#importing nltk libraries
import string
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score, train_test_split


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import keras 


#testing for individual input
testdf = pd.DataFrame(columns=['text'])
str1 = input("Enter the tweet: ")
#str1 = 'hi how are you'
print(testdf)
print(str1)
test = testdf.append({'text': str(str1)}, ignore_index=True)
print(test)

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

# removing the stopwords from the tweets
#print("downloading stopwords...") 
#nltk.download('stopwords')
stop = set(stopwords.words('english'))


d=[]
for s in test['text']:
    d.append([x for x in s.split() if x not in stop])
cleante=[]
for n in d:
    cleante.append(' '.join(n))        
test['newt']=cleante

print("\nafter removing stop word\n")
print(test)

#removing the urls from the tweets

url=[]
for s in cleante:
    url.append(re.findall('http[s]?://.*',s))
test['url']=url
test['url_cnt']=test['url'].apply(len)


#applying stemmming using snowballstemmer
from nltk import SnowballStemmer
stems=[]
clean2t=[]

# Function to apply stemming to a list of words
stemmer = SnowballStemmer(language='english')

stems=[]
clean2te=[]
    
for sen in cleante:
     stems.append([stemmer.stem(word) for word in sen.split()])
        
for n in stems:
    clean2te.append(' '.join(n))        
test['newt']=clean2te 

test.dropna()
print("\n\ncompleted stemming...")
print(test)

#vectorizing the text using tfid vectorizer
vectorizer = TfidfVectorizer()

Xtrain = pd.read_csv("xtrain_vectorizer.csv")

XX = vectorizer.fit(Xtrain['newt'])
X=XX.transform(Xtrain['newt'])
count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
train = pd.concat([Xtrain, count_vect_df], axis=1)
train=train.drop('newt',axis=1)
train=train.drop('url',axis=1)
train=train.drop('text',axis=1)
#XX = TfidfVectorizer(vocabulary=pickle.load(open("temp_models/vectorizer.pkl", "rb")))
#XX = pickle.load(open("temp_models/X_feature.pickle", "rb"))
#X = pickle.load(open("temp_models/X_transform.pickle", "rb"))
#Y = pickle.load(open("temp_models/Y_feature.pickle", "rb"))

Y=XX.transform(test['newt'])
count_vect_df = pd.DataFrame(Y.todense(), columns=vectorizer.get_feature_names())
test = pd.concat([test, count_vect_df], axis=1)
test=test.drop('newt',axis=1)
test=test.drop('url',axis=1)
test=test.drop('text',axis=1)

test.dropna()
print("\nvectorization completed...\n")
#print(test)

test.to_csv("preprocessed_test.csv")



#restoring

import os
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np

x_test = pd.read_csv("preprocessed_test.csv")
#print(x_test.head())
#print(x_test.columns)

x_test = x_test.drop(["Unnamed: 0"],axis=1)
x_test.dropna()

# Load the Model back from file
with open("mymodel", 'rb') as file:  
    model = pickle.load(file)

new_pred = model.predict(x_test)
#print(new_pred)

c = np.where(new_pred > 0,"This is a Disaster related tweet","This is a Non-Disaster tweet")
print("\n\n")
print(c)
print("\n\n")

