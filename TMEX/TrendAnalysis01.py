import nltk.tokenize
from nltk.corpus import stopwords
import json
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
import DataRead

data_files = DataRead.data_files
stop_set = set(stopwords.words('english'))

def good_word_crit(word, doc_index, tfidf_voc, tfidf_array):
	"""
	Checks if the word in document is a meaningful word or not with tfidf.
	The word and document is specified, and the tfidf information is given
	for lookup.
	tfidf_array is document_count(rows) x vocabulary_size(columns), so the 
	tfidf value we are looking for is:
	
	tfidf[doc_index, word_index]

	The index of a word can be found by tfidf_voc[word] since it is a dictionary.

	word: string type. The word that we want to check if it's significant.
	doc_index: The document constrain we want to check with the word.
	tfidf_voc: The vocabulary to lookup the word in, so we can get word index.
	tfidf_array: where the tfidf value is assorted.
	"""
	
	is_stop = word in stop_set
	is_alpha = word.isalpha()
	try:
		vec_ind = tfidf_voc[word]
	except:
		print("Failed vocabulary search: ", word)
		return False
	is_good = tfidf_array[doc_index, vec_ind]>0.1
	return (not is_stop and is_alpha and is_good)


for corpus in range(8):

	df = data_files[corpus]
	v = TfidfVectorizer(tokenizer = nltk.tokenize.word_tokenize)

	for doc_no in range(len(df.loc[:])):
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("N.K","NorthKorea")
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("NK","NorthKorea")
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("North Korea","NorthKorea")
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].lower()
		
		
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("N.K","NorthKorea")
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("NK","NorthKorea")
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("North Korea","NorthKorea")
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].lower()

		
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no),"title"] + " " + df.loc[str(doc_no)," body"]
	
	tf_idf = v.fit_transform(df[" body"])
	tfidfarray = tf_idf.toarray()
	voc = v.vocabulary_

	good_words = []

	for i in range(len(df.loc[:])):
		title = nltk.tokenize.word_tokenize(df.loc[str(i),"title"])
		good_words += [word for word in title if good_word_crit(word, i, voc, tfidfarray)]


print(nltk.FreqDist(good_words).most_common(50))
for i in nltk.FreqDist(good_words).most_common(50):
	print(i[0],end=", ")
print()