import pandas as pd
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import re
import pickle

test = pd.read_csv("./temp_output/count_added.csv")
# removing the stopwords from the tweets
print("downloading stopwords...") 
nltk.download('stopwords')
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

#remove urls from text
url=[]
for s in cleante:
    url.append(re.findall('http[s]?://.*',s))
test['url']=url
test['url_cnt']=test['url'].apply(len)

with open("./temp_output/cleante.pkl",'wb') as file:
    pickle.dump(cleante,file)

print(test)
test.to_csv("./temp_output/rm_stopword.csv",index=False)
