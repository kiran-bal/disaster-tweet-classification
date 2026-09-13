
#import neccessary packages
import pandas as pd
import re
import nltk

#defining the head value for no of lines for view
HEAD = 20

#read data as dataframe
df = pd.read_csv('../dataset/train.csv')
#print(df)

#extracting required data to new dataframe
train_df = df[['text','target']]
pd.options.display.max_rows

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

#remove url from tweets
print("\n\nRemoving urls from tweets\n")
def rm_url(x):
	x = re.sub('http[s]?://.*','',x)
	return x
print(train_df.head(HEAD))

#convert uppercase to lowercase to treat equal words equally
print("\n\nConverting to lowercase\n")
train_df['text'] = train_df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

print(train_df.head(HEAD))
for index_label,row_series in train_df.iterrows():
        train_df.at[index_label , 'text'] = rm_url(row_series['text'])
        #train_df.at[index_label , 'text'] = clean_numbers(row_series['text'])

# Cleaning punctuations n special characters

puncts = [',', '.', '"', ':', ')', '(', '-','!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\', '•',  '~', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
	x = str(x)
	for punct in puncts:
		if punct in x:
			x = x.replace(punct, f' ')
	return x

#cleaning numbers from the data and reducing exclamations
def clean_numbers(x):
	if bool(re.search(r'\d', x)):
		x = re.sub('[0-9]{5,}', '####', x)
		x = re.sub('[0-9]{4}', '###', x)
		x = re.sub('[0-9]{3}', '##', x)
		x = re.sub('[0-9]{2}', '#', x)
	return x

#applying above two functions
print(train_df.head(HEAD))
print("\n\nRemoving punctuations from tweets\n")

for index_label,row_series in train_df.iterrows():
	train_df.at[index_label , 'text'] = clean_text(row_series['text'])
print(train_df.head(HEAD))

print("\n\nReplacing numbers with \#  in tweets\n")

for index_label,row_series in train_df.iterrows():
	train_df.at[index_label , 'text'] = clean_numbers(row_series['text'])
print(train_df.head(HEAD))


#Expanding tweet slang shortforms
slang_abbrev_dict = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'B4N': 'Bye For Now',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': 'For What It\'s Worth',
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you',
    'ILU': 'I Love You',
    'IMHO': 'In My Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My Ass Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'OMG': 'Oh My God',
    'PITA': 'Pain In The Ass',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'QPSA?': 'Que Pasa?',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The Fuck',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait',
    '7K': 'Sick:-D Laugher'
}


def unslang(text):
	if text.upper() in slang_abbrev_dict.keys():
		return slang_abbrev_dict[text.upper()]
	else:
		return text

print("\n\nExpanding Tweet slangs\n")
for index_label,row_series in train_df.iterrows():
        train_df.at[index_label , 'text'] = unslang(row_series['text'])
print(train_df.head(HEAD))


from collections import Counter
import gensim
import heapq
from operator import itemgetter
from multiprocessing import Pool

'''
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', 
                                                        binary=True)
words = model.index2word
'''

#expanding contractions in the data
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
	contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
	return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
	def replace(match):
		return contractions[match.group(0)]
	return contractions_re.sub(replace, text)

#applying function to expand contractions
print("\n\nExpanding contractions in tweets\n")

for index_label,row_series in train_df.iterrows():
        train_df.at[index_label , 'text'] = replace_contractions(row_series['text'])
print(train_df.head(HEAD))


#Removing stop words from tweets
#sometimes it will depend too
print("\n\nRemoving stopwords from  tweets\n")
from nltk.corpus import stopwords
#stop = stopwords.words('english')
#train_df['text'] = train_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#nltk.download('stopwords')
stop = set(stopwords.words('english'))

#Correct misspellings using textblob
print("\n\nCorrect misspellings in tweets\n")

from textblob import TextBlob
train_df['text'][:20].apply(lambda x: str(TextBlob(x).correct()))
print(train_df.head(HEAD))

'''
#lemmatization
print("\n\nAfter lemmatization\n")
from textblob import Word
train_df['text'] = train_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(train_df['text'].head(HEAD))
'''

#tockenizing the tweets
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
def token_twt(txt):
	txt1 = tknzr.tokenize(txt)
	return txt1

for index_label,row_series in train_df.iterrows():
        train_df.at[index_label , 'text'] = token_twt(row_series['text'])
print(train_df.head(HEAD))



#Advanced preprocessing

