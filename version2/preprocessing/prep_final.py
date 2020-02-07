import pandas as pd
import numpy as np
import string
from time import time

#including my preprocessing functions
from my_prep_lib import *

string1 = "AFRICA,#AFRICANBAZE: asap goaaaaal @Breaking !!!! news:Nigeria oooooooh :-D aren't flag???? wont set ablaze..... 12000 in Aba. http://t.co/2nndBGwyEi,1"
print(string1)

print("\nReplacing url from tweet\n")
text = replaceURL(string1)
print(text)

print("\nReplacing atuser from tweet\n")
text = replaceAtUser(text)
print(text)

print("\nremoving hashtag in tweet\n")
text = removeHashtag(text)
print(text)

print("\nreplace at user in tweet\n")
text = replaceAtUser(text)
print(text)


print("\nremoving stopwords\n")
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

d=[]
d.append([x for x in text.split() if x not in stop])
d = d[0]
text = ' '.join(d)
print(text)


print("\nexpanding slangs in tweet\n")
text = replaceSlang(text)
print(text)

print("\nreplace contractions in tweet\n")
text = replaceContraction(text)
print(text)

print("\nremove numbers from tweet\n")
text = removeNumbers(text)
print(text)

print("\nremove emoticons from tweet\n")
text = removeEmoticons(text)
print(text)

#couting multple punctuations
print("\ncounting multiple punctuations\n")
MultiExclMarks = 0
MultiQuesMarks = 0
MultiStopMarks = 0

MultiExclMarks += countMulExcl(text)
MultiQuesMarks += countMulQues(text)
MultiStopMarks += countMulStop(text)

print(MultiExclMarks,MultiQuesMarks,MultiStopMarks)

print("\nremove multiexclamations from tweet\n")
text = replaceMulExcl(text)
print(text)

print("\nremove multiquestionmarks from tweet\n")
text = replaceMulQues(text)
print(text)

print("\nremove multistopmarks from tweet\n")
text = replaceMulStop(text)
print(text)


print("\nshortening elongated words\n")
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

