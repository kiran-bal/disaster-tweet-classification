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
df = pd.read_csv("../dataset/train.csv")
final_df = []
missed = []
for index, row in df.iterrows():
	l = []
	#string1 = 'new data has found'
	print(row['text'])
	string1 = row['text']
	#l.append(string1)
	try:
		l.append(list(document_vector(w2v,string1)))
		print(l)
	except:
		l = [0.69671315, 0.049782764, -0.24523668, -0.15872465, -0.0665417, 0.20241983, 0.077576466, 1.9189811, -0.006817713, 0.06951927, -0.4152605, -0.89838743, -3.7327635, -0.049629666, -0.7617386, -0.5561758, -0.9451503, 0.035365578, -1.0393411, 0.2922259, -0.16664228, -0.46666014, 0.35039356, 0.40681368, 0.38142973]
		missed.append(str(index) + " ")
	#a_series = pd.Series(l)
	new_df = new_df.append(l, ignore_index=True)

	#new_df.append(l)
	print("\n")
print(new_df)
print(missed)
new_df.to_csv("w2v_train_data.csv",index=False)
	
#list1 = []
#list1.append(document_vector(string1))
#print(list1)
