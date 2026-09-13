import pandas as pd
import gensim
from k_prep import *
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
df = pd.read_csv("../dataset/train.csv")
df['text'] = df['text'].apply(k_prep)
print(df['text'])


with open("lstm_model_backup", 'rb') as file:  
    model = pickle.load(file)

model.load_weights('./best_model2_16bz_preprocessed.h5')

test = pd.read_csv('../dataset/test.csv')
test_X = test.text.values
print(test_X)


maxlen = 100
max_features = 10000

with open("tokenizer_backup", 'rb') as file:  
    tokenizer = pickle.load(file)


test_X = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(test_X, maxlen=maxlen)

probabilities = model.predict(test_X)
print(probabilities)

