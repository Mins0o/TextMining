import json
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import hdbscan
import os
import pickle

pre_path = "."
data_path = pre_path + "/data/koreaherald_1517_"
result_dir = "./results/"
result_file_name = result_dir + "Titling.txt"

# The embeddings of the documents can be saved with pickle
check_points_dir = "./checkpoints/"
embedding_file_name = "embeddings.mdl"
umap_embd_file_name = check_points_dir+"umap_"+embedding_file_name
cluster_file_name = check_points_dir+"cluster_"+embedding_file_name
embedding_file_name = check_points_dir+embedding_file_name


data_files=[]

# load data into list:data_files
for corpus in range(8):
	with open( data_path + str(corpus) + ".json", 'r') as f:
		data=json.load(f)
	df = pd.DataFrame.from_dict(data)

  # Preprocess N.K
	# for doc_no in range(len(df.loc[:])):
	# 	df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("N.K ","NorthKorea ")
	# 	df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("N. Korea","NorthKorea ")
	# 	df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("NK ","NorthKorea ")
	# 	df.loc[str(doc_no),"title"] = df.loc[str(doc_no),"title"].replace("North Korea ","NorthKorea ")
		
	# 	df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("N.K ","NorthKorea ")
	# 	df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("N. Korea ","NorthKorea ")
	# 	df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("NK ","NorthKorea ")
	# 	df.loc[str(doc_no)," body"] = df.loc[str(doc_no)," body"].replace("North Korea ","NorthKorea ")
	data_files.append(df)


# all_docs_in_list stores each document bodies, headlines, and dates as [string,,] object in list
all_docs_in_list = []

for i in data_files:
  all_docs_in_list += i.loc[:,[" body","title"," time"]].values.tolist()

# we are calling the document bodies 'data'
data = [i[0] for i in all_docs_in_list]

# Doc2Vec

model = SentenceTransformer('paraphrase-mpnet-base-v2')
if os.path.isfile(embedding_file_name):
  print(">> Using previous embeddings")
  with open(embedding_file_name,"rb") as embed_model:
    embeddings = pickle.load(embed_model)
else:
  print(">> Generating new embeddings")
  embeddings = model.encode(data, show_progress_bar=True)
  with open(embedding_file_name,"wb") as embed_model:
    pickle.dump(embeddings,embed_model)

def c_tf_idf(documents, m, ngram_range=(4, 6)):
    """Class-based TF-IDF: Used BERTopic"""
    #vectorized = tfidfVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    vectorized = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = vectorized.transform(documents).toarray()
    print("calculating")
    tf_idf = np.multiply(np.divide(t.T, t.sum(axis=1)), np.log(np.divide(m, t.sum(axis=0))).reshape(-1, 1))

    return tf_idf, vectorized

def extract_top_n_words_per_topic(tf_idf, count, per_topic, n=20):
    words = count.get_feature_names()
    labels = list(per_topic.Topic)
    # print(len(tf_idf))
    tf_idf_transposed = tf_idf.T
    # print(len(tf_idf_transposed))
    #sort by tf_idf score
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

with open(result_file_name,'w') as result_txt:
  # Reduce dimension and Cluster
  print("Reducing dimension")
  if os.path.isfile(umap_embd_file_name):
    print(">> Using previous reduction")
    with open(umap_embd_file_name,"rb") as umap_embeds:
      umap_embeddings = pickle.load(umap_embeds)
  else:
    print(">> Reducing embeddings again")
    umap_embeddings = umap.UMAP(n_neighbors=20, n_components=7, min_dist = 0.02, metric='cosine').fit_transform(embeddings)  
    with open(umap_embd_file_name,"wb") as umap_embeds:
      pickle.dump(umap_embeddings,umap_embeds)
  
  print("Clustering")
  if os.path.isfile(cluster_file_name):
    print(">> Using previous clusters")
    with open(cluster_file_name,"rb") as cluster_file:
      cluster = pickle.load(cluster_file)
  else:
    print(">> Clustering embeddings again")
    cluster = hdbscan.HDBSCAN(min_cluster_size=45, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
    with open(cluster_file_name,"wb") as cluster_file:
      pickle.dump(cluster,cluster_file)
  
  # constructing a data frame:
  # Rows  |Docs(text body)  |Doc_ID |Topic(labels)  |Title(headline) |Date
  #       |                 |       |               |                |
  docs_df = pd.DataFrame(data, columns=["Doc"])
  docs_df['Doc_ID'] = range(len(docs_df))
  docs_df['Topic'] = cluster.labels_
  docs_df["Title"] = [i[1] for i in all_docs_in_list]
  docs_df["Date"] = [i[2] for i in all_docs_in_list]

  if(True): 
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    
    #scoring ngrams in the collections
    print("scoring ngrams in the collections")
    d_tf_idf, d_count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))

    # Now all the documents are clustered
    # Extract_top_n_words_per_topic(tf_idf, count, per_topic, n=20)
    # From the text bodies
    d_top_n_ngrams = extract_top_n_words_per_topic(d_tf_idf, d_count, docs_per_topic, n=20)

    # Count how much is in each topics
    topic_sizes = extract_topic_sizes(docs_df)
    print("Number of Clusters: ",len(topic_sizes))
    result_txt.write(str(len(topic_sizes))+"\n")
    top_tens = topic_sizes.head(10)
    print(top_tens)
    result_txt.write(top_tens.to_string()+"\n")

    # Print out the top_ten topics' top n ngrams
    for topic_label in top_tens.loc[:,"Topic"]:
      print(topic_label,end="\t")
      print(d_top_n_ngrams[i])
      result_txt.write("{}\t".format(i))
      result_txt.write(d_top_n_ngrams[i].__repr__())
      result_txt.write("\n")
    result_txt.write("\n")
    
    for topic_label in top_tens.loc[:,"Topic"]:
      print(topic_label,end="\t")
      print(top_tens.loc[topic_label+1,"Size"],end="\t")
      print(d_top_n_ngrams[i][:2])
      result_txt.write("{}\t".format(i))
      result_txt.write(topic_label.loc[i+1,"Size"].__repr__()+"\t")
      for j in d_top_n_ngrams[i][:2]:
        result_txt.write(j[0]+"\t")
      result_txt.write("\n")

  if(True):
    titles_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Title': ' '.join})
    #scoring ngrams in the collections
    print("scoring ngrams in the collections")
    t_tf_idf, t_count = c_tf_idf(titles_per_topic.Title.values, m=len(data))
    # Now all the documents are clustered
    # Extract_top_n_words_per_topic(tf_idf, count, per_topic, n=20)
    # From the headlines
    t_top_n_ngrams = extract_top_n_words_per_topic(t_tf_idf, t_count, titles_per_topic, n=20)

    # Print out the top_ten topics' top n ngrams
    for topic_label in top_tens.loc[:,"Topic"]:
      print(topic_label,end="\t")
      print(t_top_n_ngrams[i])
      result_txt.write("{}\t".format(i))
      result_txt.write(t_top_n_ngrams[i].__repr__())
      result_txt.write("\n")
    result_txt.write("\n")
    
    for topic_label in top_tens.loc[:,"Topic"]:
      print(topic_label,end="\t")
      print(top_tens.loc[topic_label+1,"Size"],end="\t")
      print(t_top_n_ngrams[i][:2])
      result_txt.write("{}\t".format(i))
      result_txt.write(topic_label.loc[i+1,"Size"].__repr__()+"\t")
      for j in t_top_n_ngrams[i][:2]:
        result_txt.write(j[0]+"\t")
      result_txt.write("\n")

#np.linalg.norm(umap_embeddings[702]-umap_embeddings[1734])