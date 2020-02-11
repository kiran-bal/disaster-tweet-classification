#generating word2vec embedding from Glove
print("\nCreating word2vec embeddings...\n")
import pandas as pd
import numpy as np

'''
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = '../../../temp/glove/glove.twitter.27B.25d.txt'
word2vec_output_file = 'glove.twitter.27B.25d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
'''

def document_vector(model,doc):
	#filename = 'glove.twitter.27B.25d.txt.word2vec'
	#model = KeyedVectors.load_word2vec_format(filename, binary=False)
	# remove out-of-vocabulary words
	doc = [word for word in doc if word in model.vocab]
	return np.mean(model[doc], axis=0)


from gensim.models import KeyedVectors
filename = 'glove.twitter.27B.25d.txt.word2vec'
print("\n Converting...")
w2v = KeyedVectors.load_word2vec_format(filename, binary=False)
#print(w2v)
#word_doc = [word for word in word_doc if word in w2v.vocab]

new_df = pd.DataFrame()
df = pd.read_csv("../dataset/test.csv")

for i in range(30):
	l = []
	#string1 = 'new data has found'
	print(df['text'][i])
	string1 = df['text'][i]
	l.append(string1)
	l.append(list(document_vector(w2v,string1)))
	print(l)
	new_df.append(l)
	print("\n")
print(new_df)
	
#list1 = []
#list1.append(document_vector(string1))
#print(list1)
