import pandas as pd
import numpy as np
import string
from time import time
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

#including my preprocessing functions
from my_prep_lib import *

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


def k_prep(inputText = 'null',c=0):
	
	if c == "2":
		text = inputText
	else:
		c = input("\n1.Enter or 2.default : ")
		if c == "2":
			text = "AFRICA,#AFRICANBAZE: asap goaaaaal @Breaking !!!! news:Nigeria oooooooh :-D aren't flag???? wont set ablaze..... 12000 in America. http://t.co/2nndBGwyEi,1"
			print(text)
		else:
			text = input("\nEnter the tweet: ")
	
	#text=inputText
	print("\nRemove unicode from tweet")
	text = removeUnicode(text)
	print(text)

	print("\nReplacing url from tweet")
	text = replaceURL(text)
	print(text)

	print("\nReplacing atuser from tweet")
	text = replaceAtUser(text)
	print(text)

	print("\nremoving hashtag in tweet")
	text = removeHashtag(text)
	print(text)

	print("\nreplace at user in tweet")
	text = replaceAtUser(text)
	print(text)

	'''
	print("\nremoving stopwords")
	#nltk.download('stopwords')
	from nltk.corpus import stopwords
	stop = set(stopwords.words('english'))

	d=[]
	d.append([x for x in text.split() if x not in stop])
	d = d[0]
	text = ' '.join(d)
	print(text)
	'''

	print("\nremove numbers from tweet")
	text = removeNumbers(text)
	print(text)

	print("\nremove emoticons from tweet")
	text = removeEmoticons(text)
	print(text)

	#couting multple punctuations
	print("\ncounting multiple punctuations")
	MultiExclMarks = 0
	MultiQuesMarks = 0
	MultiStopMarks = 0

	MultiExclMarks += countMulExcl(text)
	MultiQuesMarks += countMulQues(text)
	MultiStopMarks += countMulStop(text)

	print(MultiExclMarks,MultiQuesMarks,MultiStopMarks)

	print("\nremove multiexclamations from tweet")
	text = replaceMulExcl(text)
	print(text)

	print("\nremove multiquestionmarks from tweet")
	text = replaceMulQues(text)
	print(text)

	print("\nremove multistopmarks from tweet\n")
	text = replaceMulStop(text)
	print(text)


	print("\nshortening elongated words")
	totalElongated = 0
	totalElongated += countElongated(text)
	print(totalElongated)

	regex1 = re.compile(r"(.)\1{2}")
	l=[]
	for word in text.split():
		if(regex1.search(word)):
			new_word = replaceElongated(word)
			#print(new_word)
			l.append(new_word)
		else:
			l.append(word)
	text = ' '.join(l)
	print(text)

	print("\nRemoving punctuations except ?!")
	text = removePuncts(text)
	print(text)
	
	print("\nexpanding slangs in tweet")
	text = replaceSlang(text)
	print(text)

	print("\nreplace contractions in tweet")
	text = replaceContraction(text)
	print(text)
	
	print("\nTokenizing the text")
	text = word_tokenize(text)
	
	print("\nLemmatizing the text")
	lemma = WordNetLemmatizer()
	
	list1 = []
	for txt in text:
		list1.append(lemma.lemmatize(txt))
	
	print(list1)
	return list1	
'''
def document_vector(doc):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in w2v.wv.vocab]
    return np.mean(w2v[doc], axis=0)


def myword2vec():
	#generating word2vec embedding from Glove
	print("\nCreating word2vec embeddings...\n")

	
	from gensim.scripts.glove2word2vec import glove2word2vec
	glove_input_file = '../../../temp/glove/glove.twitter.27B.100d.txt'
	word2vec_output_file = 'glove.twitter.27B.100d.txt.word2vec'
	glove2word2vec(glove_input_file, word2vec_output_file)
	
	
	#(above code for first run to convert glove)
	from gensim.models import KeyedVectors
	filename = 'glove.twitter.27B.100d.txt.word2vec'
	w2v = KeyedVectors.load_word2vec_format(filename, binary=False)
	print("\n Converting...")
	print(w2v)
	#word_doc = [word for word in word_doc if word in w2v.vocab]
	print(np.mean(w2v[word_doc], axis=0))
	'''
	
'''
df = pd.read_csv("../dataset/train.csv")
df['text'] = df['text'].apply(k_prep)
print(df['text'])
'''
k_prep()
