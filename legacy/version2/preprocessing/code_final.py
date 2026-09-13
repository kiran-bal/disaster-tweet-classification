import pandas as pd
from prep_final import *
import gensim

df = pd.read_csv("../dataset/train.csv")
'''
for i in range(2):
	print(df['text'][i+1])
	result = k_prep(df['text'][i+1])
	print(result)
	print("\n")

myword2vec(result)
'''
myword2vec()
#w2v = gensim.models.Word2Vec(result, size=350, window=10, min_count=2, iter=20)

