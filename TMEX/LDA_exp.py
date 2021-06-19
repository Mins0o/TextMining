import DataRead
import gensim
import nltk
from nltk.stem import PorterStemmer


data_files = DataRead.data_files

stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(nltk.WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


all_docs = []

for df in data_files[:1]:
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

	processed_docs = list(df.loc[:," body"])
	processed_docs = [preprocess(text) for text in processed_docs]
	all_docs += processed_docs

dictionary = gensim.corpora.Dictionary(all_docs)

bow_corpus = [dictionary.doc2bow(doc) for doc in all_docs]

lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=11, id2word = dictionary,passes = 10)

print(lda_model.show_topics(num_topics=11))
for i in lda_model.show_topics(num_topics=11):
	print(i)
lda_model.save("test1.model")