import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # importing tensorflow library
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os
import datetime

#importing nltk libraries
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import Lasso, SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import keras


#import the training data
train_df = pd.read_csv("dataset/train.csv")
print(train_df)

#import the test data
test_df = pd.read_csv("dataset/test.csv")
print(test_df)

#printing the target tweets
dis_twts = train_df[train_df["target"] == 1]
non_dis_twts = train_df[train_df["target"] == 0]

print("\nDisater Tweets\n")
print(dis_twts)
print("\nNon Disater Tweets\n")
print(non_dis_twts)


df = train_df[['text','target']]
test = test_df[['text']]
test_save = test_df[['text']]

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
df['no_chars'] = df['text'].apply(len)
df['no_words'] = df['text'].str.split().apply(len)
df['no_sent'] = df['text'].str.split('.').apply(len)
df['no_para'] = df['text'].str.split('\n').apply(len)
df['avg_word_len'] = df['text'].apply(avg_word_len)
df['no_hashtags'] = df['text'].apply(hash_count)
df['no_mentions'] = df['text'].apply(mention_count)

test['no_chars'] = test['text'].apply(len)
test['no_words'] = test['text'].str.split().apply(len)
test['no_sent'] = test['text'].str.split('.').apply(len)
test['no_para'] = test['text'].str.split('\n').apply(len)
test['avg_word_len'] = test['text'].apply(avg_word_len)
test['no_hashtags'] = test['text'].apply(hash_count)
test['no_mentions'] = test['text'].apply(mention_count)

print("\nafter calculating counts\n")
print(df)

# removing the stopwords from the tweets
print("downloading stopwords...") 
#nltk.download('stopwords')
stop = set(stopwords.words('english'))

d=[]
for s in df['text']:
    

       d.append([x for x in s.split() if x not in stop])
cleant=[]
for n in d:
    cleant.append(' '.join(n))        
df['newt']=cleant



d=[]
for s in test['text']:
    d.append([x for x in s.split() if x not in stop])
cleante=[]
for n in d:
    cleante.append(' '.join(n))        
test['newt']=cleante

print("\nafter removing stop word\n")
print(df)

#removing the urls from the tweets
url=[]
for s in cleant:
    url.append(re.findall('http[s]?://.*',s))
df['url']=url
df['url_cnt']=df['url'].apply(len)
df.sample(5)

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
for sen in cleant:
     stems.append([stemmer.stem(word) for word in sen.split()])
        
for n in stems:
    clean2t.append(' '.join(n))        
df['newt']=clean2t
        


stems=[]
clean2te=[]
    
for sen in cleante:
     stems.append([stemmer.stem(word) for word in sen.split()])
        
for n in stems:
    clean2te.append(' '.join(n))        
test['newt']=clean2te 

print("\n\ncompleted stemming...")
print(df)

Xtrain=df.drop('target',axis=1)
ytrain=df['target']

print("\nxtrain data\n")
print(Xtrain)
#Xtrain.to_csv("temp/xtrain_vectorizer.csv")
#vectorizing the text using tfid vectorizer
vectorizer = TfidfVectorizer()
XX = vectorizer.fit(Xtrain['newt'])

#print("\nsaving the vectorixer model\n")
#pickle.dump(vectorizer.vocabulary_,open("temp_models/vectorizer.pkl","wb"))


X=XX.transform(Xtrain['newt'])
count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
train = pd.concat([Xtrain, count_vect_df], axis=1)
train=train.drop('newt',axis=1)
train=train.drop('url',axis=1)
train=train.drop('text',axis=1)

print("\nsaving the vectorixer model\n")
#pickle.dump(vectorizer.vocabulary_,open("temp_models/vectorizer.pkl","wb"))

Y=XX.transform(test['newt'])
count_vect_df = pd.DataFrame(Y.todense(), columns=vectorizer.get_feature_names())
test = pd.concat([test, count_vect_df], axis=1)
test=test.drop('newt',axis=1)
test=test.drop('url',axis=1)
test=test.drop('text',axis=1)

#saving the vectorizer
#pickle.dump(XX, open("temp_models/X_feature.pickle", "wb"))
#pickle.dump(X, open("temp_models/X_transform.pickle", "wb"))
#pickle.dump(Y, open("temp_models/Y_feature.pickle", "wb"))

print("\nafter vectorization\n")
print(df)

#splitting the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(train,ytrain, test_size=0.1)

#save temporary files for restoring
'''
X_test.to_csv("temp/X_test_data.csv")
y_train.to_csv("temp/y_train_data.csv")
y_test.to_csv("temp/y_test_data.csv")
X_train.to_csv("temp/X_train_data.csv")
test_save.to_csv("temp/text_data_test.csv")
'''

#COMPARISON OF MODELS

#initialize models

ridge = RidgeClassifier(alpha=.8,solver='sag')
lass = Lasso(alpha=.01)
rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=100)
#svm = LinearSVC(penalty='l2',max_iter=100)
svm = SVC(kernel='linear',max_iter=100)
nb = GaussianNB()
nb2 = MultinomialNB()
dtc = DecisionTreeClassifier()
logreg = LogisticRegression()
sgd = SGDClassifier()
gbc = GradientBoostingClassifier()

#X_train.flags
#y_train.flags
#prediction

'''
print("\n\nRidge classifier\n")
ridge.fit(X_train,y_train)
Y_pred = ridge.predict(X_test)
print(classification_report(y_test,Y_pred))
'''
print("\n\nLogistic Regression\n")
logreg.fit(X_train,y_train)
Y_pred = logreg.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))


print("\n\nLasso classifier\n")
lass.fit(X_train,y_train)
Y_pred = list(map(lambda x: x>=.5, lass.predict(X_test)))
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))



print("\n\nRandom forest classifier\n")
rf.fit(X_train,y_train)
Y_pred = rf.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))


print("\n\nGaussian Bayes classifier\n")
nb.fit(X_train,y_train)
Y_pred = nb.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))

print("\n\nMultinomial Bayesian classifier\n")
nb2.fit(X_train,y_train)
Y_pred = nb2.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))

print("\n\nDecision Tree classifier\n")
dtc.fit(X_train,y_train)
Y_pred = dtc.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))


print("\n\nSupport vector machine\n")
svm.fit(X_train,y_train)
Y_pred = svm.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))


#selecting ridgeclassifier as model
print("\n\nRidge classifier\n")
ridgemodel = RidgeClassifier()
ridgemodel.fit(X_train, y_train)

y_pred = ridgemodel.predict(X_test)

print("Accuracy Score is :")
print(accuracy_score(y_test, y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


print("\n\nStochastic Gradient Descent classifier\n")
sgd.fit(X_train,y_train)
Y_pred = sgd.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))


print("\n\Gradient Boosting classifier\n")
gbc.fit(X_train,y_train)
Y_pred = gbc.predict(X_test)
print("Accuracy Score is :")
print(accuracy_score(y_test, Y_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test,Y_pred))


# using majority voting technique for all models
print("\n\n\n majority voting technique\n")

modelnames = [('LogisticRegression',logreg),
		('SGDClassifier',sgd),
		('Gradientboosting',gbc),
		('NaiveBayes',nb),
		('MultinomialNB',nb2),
		('RidgeClassifier',ridgemodel)]

vc = VotingClassifier(voting = 'hard',estimators=modelnames)
vc.fit(X_train,y_train)
Y_pred = vc.predict(X_test)

print("Accuracy score is:")
print(accuracy_score(y_test,Y_pred))
print(classification_report(y_test,Y_pred))
print(confusion_matrix(y_test,Y_pred))
