import json
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import hdbscan

#pre_path = "."
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

for i in data_files:
  all_docs_in_list += list(i.loc[:," body"])
  all_titles_in_list += list(i.loc[:,"title"])

data = all_docs_in_list

#Doc2Vec
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)

def c_tf_idf(documents, m, ngram_range=(1, 2)):
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

with open("result_.txt",'w') as result_txt:
  for neighbor_count in range(5,45,5):
    # Reduce dimension and Cluster
    print("Neighbor Size:\t",neighbor_count)
    result_txt.write("Neighbor Size:{}\n".format(neighbor_count))
    umap_embeddings = umap.UMAP(n_neighbors=neighbor_count, n_components=7, min_dist = 0.05, metric='cosine').fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
    docs_df = pd.DataFrame(data, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_df["Title"] = all_titles_in_list
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df)
    print(len(topic_sizes))
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

  result_txt.write("\n\n")

  for n_components in range(2,30):
    # Reduce dimension and Cluster
    print("Component Numbers:\t",n_components)
    result_txt.write("Component Numbers:{}\n".format(n_components))
    umap_embeddings = umap.UMAP(n_neighbors=12, n_components=n_components, min_dist = 0.05, metric='cosine').fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
    docs_df = pd.DataFrame(data, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_df["Title"] = all_titles_in_list
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df)
    print(len(topic_sizes))
    result_txt.write(str(len(topic_sizes))+"\n")
    a = topic_sizes.head(10)
    result_txt.write(a.to_string()+"\n")
    for i in a.loc[:,"Topic"]:
      print(i,end="\t")
      print(top_n_words[i])
      result_txt.write("{}\t".format(i))
      result_txt.write(top_n_words[i].__repr__())
      result_txt.write("\n")
    result_txt.write("\n")

  result_txt.write("neighbor:20\n")
  for n_components in range(2,30):
    # Reduce dimension and Cluster
    print("Component Numbers:\t",n_components)
    result_txt.write("Component Numbers:{}\n".format(n_components))
    umap_embeddings = umap.UMAP(n_neighbors=20, n_components=n_components, min_dist = 0.05, metric='cosine').fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
    docs_df = pd.DataFrame(data, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_df["Title"] = all_titles_in_list
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df)
    print(len(topic_sizes))
    result_txt.write(str(len(topic_sizes))+"\n")
    a = topic_sizes.head(10)
    result_txt.write(a.to_string()+"\n")
    for i in a.loc[:,"Topic"]:
      print(i,end="\t")
      print(top_n_words[i])
      result_txt.write("{}\t".format(i))
      result_txt.write(top_n_words[i].__repr__())
      result_txt.write("\n")
    result_txt.write("\n")