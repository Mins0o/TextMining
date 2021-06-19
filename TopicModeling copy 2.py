import json
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import hdbscan
import os
import pickle

pre_path = "."
data_path = pre_path + "/data/koreaherald_1517_"

data_files=[]

for corpus in range(8):
	with open( data_path + str(corpus) + ".json", 'r') as f:
		data=json.load(f)
	df = pd.DataFrame.from_dict(data)
	for doc_no in range(len(df.loc[:])):
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("N.K ","NorthKorea ")
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("N. Korea ","NorthKorea ")
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("NK ","NorthKorea ")
		df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("North Korea ","NorthKorea ")
		
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("N.K ","NorthKorea ")
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("N. Korea ","NorthKorea ")
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("NK ","NorthKorea ")
		df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("North Korea ","NorthKorea ")
	data_files.append(df)





all_docs_in_list = []
all_titles_in_list = []
y2014 = [i[(i[" time"]<"2015") & (i[" time"]>"2014")] for i in data_files if i[(i[" time"]<"2015") & (i[" time"]>"2014")].size != 0 ]
y2015 = [i[(i[" time"]<"2016") & (i[" time"]>"2015")] for i in data_files if i[(i[" time"]<"2016") & (i[" time"]>"2015")].size != 0 ]
y2016 = [i[(i[" time"]<"2017") & (i[" time"]>"2016")] for i in data_files if i[(i[" time"]<"2017") & (i[" time"]>"2016")].size != 0 ]
y2017 = [i[(i[" time"]<"2018") & (i[" time"]>"2017")] for i in data_files if i[(i[" time"]<"2018") & (i[" time"]>"2017")].size != 0 ]
y2018 = [i[(i[" time"]<"2019") & (i[" time"]>"2018")] for i in data_files if i[(i[" time"]<"2019") & (i[" time"]>"2018")].size != 0 ]
y2019 = [i[(i[" time"]<"2020") & (i[" time"]>"2019")] for i in data_files if i[(i[" time"]<"2020") & (i[" time"]>"2019")].size != 0 ]

y_2014 = []
y_2015 = []
y_2016 = []
y_2017 = []
y_2018 = []
y_2019 = []
y_2014t = []
y_2015t = []
y_2016t = []
y_2017t = []
y_2018t = []
y_2019t = []

for i in y2014:
  y_2014  += list(i.loc[:," body"])
  y_2014t += list(i.loc[:,"title"])
for i in y2015:
  y_2015  += list(i.loc[:," body"])
  y_2015t += list(i.loc[:,"title"])
for i in y2016:
  y_2016  += list(i.loc[:," body"])
  y_2016t += list(i.loc[:,"title"])
for i in y2017:
  y_2017  += list(i.loc[:," body"])
  y_2017t += list(i.loc[:,"title"])
for i in y2018:
  y_2018  += list(i.loc[:," body"])
  y_2018t += list(i.loc[:,"title"])
for i in y2019:
  y_2019  += list(i.loc[:," body"])
  y_2019t += list(i.loc[:,"title"])

ys = [y_2014,y_2015,y_2016,y_2017,y_2018,y_2019]
ys_t = [y_2014t,y_2015t,y_2016t,y_2017t,y_2018t,y_2019t]
def c_tf_idf(documents, m, ngram_range=(4, 6)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes
year_n = 2014
for year_nn in range(len(ys)):
  print(year_n)
  #Doc2Vec
  year = ys[year_nn]
  if not len(year)==0:
    all_titles_in_list = ys_t[year_nn]
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    print("generating new embeddings")
    embeddings = model.encode(year, show_progress_bar=True)



    with open("Titling"+str(year_n)+".txt",'w') as result_txt:
      result_txt.write(str(year_n)+"\n")
      # Reduce dimension and Cluster
      print("Reducing dimension")
      umap_embeddings = umap.UMAP(n_neighbors=15, n_components=4, min_dist = 0.02, metric='cosine').fit_transform(embeddings)
      print("Clustering")
      cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
      docs_df = pd.DataFrame(year, columns=["Doc"])
      docs_df['Topic'] = cluster.labels_
      docs_df['Doc_ID'] = range(len(docs_df))
      docs_df["Title"] = all_titles_in_list
      docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Title': ' '.join})
      tf_idf, count = c_tf_idf(docs_per_topic.Title.values, m=len(year))
      top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
      topic_sizes = extract_topic_sizes(docs_df)
      print("Number of Clusters: ",len(topic_sizes))
      result_txt.write(str(len(topic_sizes))+"\n")
      a = topic_sizes.head(10)
      print(a)
      result_txt.write(a.to_string()+"\n")
      for i in a.loc[:,"Topic"]:
        print(i,end="\t")
        print(top_n_words[i])
        result_txt.write("{}\t".format(i))
        result_txt.write(top_n_words[i].__repr__())
        result_txt.write("\n")
      result_txt.write("\n")
      
      for i in a.loc[:,"Topic"]:
        print(i,end="\t")
        print(a.loc[i+1,"Size"],end="\t")
        print(top_n_words[i][:2])
        result_txt.write("{}\t".format(i))
        result_txt.write(a.loc[i+1,"Size"].__repr__()+"\t")
        for j in top_n_words[i][:2]:
          result_txt.write(j[0]+"\t")
        result_txt.write("\n")
  year_n +=1
