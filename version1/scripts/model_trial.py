




vectorizer = CountVectorizer(decode_error="replace")
vec_train = vectorizer.fit_transform(corpus)
#Save vectorizer.vocabulary_
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

#Load it later
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))
