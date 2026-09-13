import pandas as pd
import re
import pickle

with open("./temp_output/cleante.pkl","rb") as file:
    cleante = pickle.load(file)

test = pd.read_csv("./temp_output/rm_stopword.csv")
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
test.to_csv("./temp_output/stemmed.csv",index=False)
