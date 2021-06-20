print("importing modules      ", end = "\r")
import json
import pandas as pd

import numpy as np
print("importing vectorizer    ", end = "\r")
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

import os
import pickle

pre_path = "."
data_path = pre_path + "/data/koreaherald_1517_"
result_dir = "./results/"
t_result_file_name = result_dir + "Titling_t.txt"
d_result_file_name = result_dir + "Titling_d.txt"

# The embeddings of the documents can be saved with pickle
check_points_dir = "./checkpoints/"
embedding_file_name = "embeddings3.mdl"
umap_embd_file_name = check_points_dir+"umap_"+embedding_file_name
cluster_file_name = check_points_dir+"cluster_"+embedding_file_name
embedding_file_name = check_points_dir+embedding_file_name


data_files=[]

# load data into list:data_files
print("...loading data...     ", end = "\r")
for corpus in range(8):
	with open( data_path + str(corpus) + ".json", 'r') as f:
		data=json.load(f)
	df = pd.DataFrame.from_dict(data)
	data_files.append(df)


# all_docs_in_list stores each document bodies, headlines, and dates as [string,,] object in list
all_docs = data_files[0].loc[:,[" body","title"," time"]]

for i in data_files[1:]:
  #all_docs_in_list += i.loc[:,[" body","title"," time"]].values.tolist()
  all_docs = pd.concat([all_docs,i.loc[:,[" body","title", " time"]]],sort = False)

# we are calling the document bodies 'data'
data = list(all_docs.loc[:," body"])

# Doc2Vec

if os.path.isfile(embedding_file_name):
  print(">> Using previous embeddings")
  with open(embedding_file_name,"rb") as embed_model:
    embeddings = pickle.load(embed_model)
else:
  print(">> Generating new embeddings")
  model = SentenceTransformer('paraphrase-mpnet-base-v2')
  embeddings = model.encode(data, show_progress_bar=True)
  with open(embedding_file_name,"wb") as embed_model:
    pickle.dump(embeddings,embed_model)

def c_tf_idf(documents, m, ngram_range=(4, 5)):
    vectorized = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = vectorized.transform(documents).toarray()
    tf = np.divide(t.T, t.sum(axis=1))
    idf = np.log(np.divide(m, t.sum(axis=0))).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
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

# Reduce dimension and Cluster
print("Reducing dimension")
if os.path.isfile(umap_embd_file_name):
  print(">> Using previous reduction")
  with open(umap_embd_file_name,"rb") as umap_embeds:
    umap_embeddings = pickle.load(umap_embeds)
else:
  print(">> Reducing embeddings again")
  import umap
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
  import hdbscan
  cluster = hdbscan.HDBSCAN(min_cluster_size=45, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
  with open(cluster_file_name,"wb") as cluster_file:
    pickle.dump(cluster,cluster_file)

# constructing a data frame:
# Rows  |Doc(text body)  |Doc_ID |Topic(labels)  |Title(headline) |Date
#       |                |       |               |                |
docs_df = pd.DataFrame(data, columns=["Doc"])
docs_df['Doc_ID'] = range(len(docs_df))
docs_df['Topic'] = cluster.labels_
docs_df["Title"] = list(all_docs.loc[:,"title"])
docs_df["Date"] = list(all_docs.loc[:," time"])


if(True): 
  print("\n\nFrom Headlines\n\n")
  with open(t_result_file_name,'w') as result_txt:
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Title': ' '.join})
    
    #scoring ngrams in the collections
    print("scoring ngrams in the collections")
    t_tf_idf, t_count = c_tf_idf(docs_per_topic.Title.values, m=len(data))

    # Now all the documents are clustered
    # Extract_top_n_words_per_topic(tf_idf, count, per_topic, n=20)
    # From the text bodies
    t_top_n_ngrams = extract_top_n_words_per_topic(t_tf_idf, t_count, docs_per_topic, n=20)

    # Count how much is in each topics
    topic_sizes = extract_topic_sizes(docs_df)
    print("Number of Clusters: ",len(topic_sizes))
    result_txt.write(str(len(topic_sizes))+"\n")
    top_tens = topic_sizes.head(20)
    print(top_tens)
    result_txt.write(top_tens.to_string()+"\n")

    # Print out the top_ten topics' top n ngrams
    for topic_label in top_tens.loc[:,"Topic"]:
      result_txt.write("{}\t".format(topic_label))
      result_txt.write(t_top_n_ngrams[topic_label].__repr__())
      result_txt.write("\n")
    result_txt.write("\n")
    
    for topic_label in top_tens.loc[:,"Topic"]:
      print(topic_label,end="\t")
      print(top_tens.loc[topic_label+1,"Size"],end="\t")
      print(t_top_n_ngrams[topic_label][:3])
      result_txt.write("{}\t".format(topic_label))
      result_txt.write(top_tens.loc[topic_label+1,"Size"].__repr__()+"\t")
      for j in t_top_n_ngrams[topic_label][:3]:
        result_txt.write(j[0]+"\t")
      result_txt.write("\n")

if(False):
  print("\n\nFrom Text bodies\n\n")
  with open(d_result_file_name,'w') as result_txt:
    docs_df = docs_df[:len(docs_df)//2]
    titles_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    #scoring ngrams in the collections
    print("scoring ngrams in the collections")
    d_tf_idf, d_count = c_tf_idf(titles_per_topic.Doc.values, m=len(data))
    # Now all the documents are clustered
    # Extract_top_n_words_per_topic(tf_idf, count, per_topic, n=20)
    # From the headlines
    d_top_n_ngrams = extract_top_n_words_per_topic(d_tf_idf, d_count, titles_per_topic, n=20)

    # Print out the top_ten topics' top n ngrams
    for topic_label in top_tens.loc[:,"Topic"]:
      result_txt.write("{}\t".format(topic_label))
      result_txt.write(d_top_n_ngrams[topic_label].__repr__())
      result_txt.write("\n")
    result_txt.write("\n")
    
    for topic_label in top_tens.loc[:,"Topic"]:
      print(topic_label,end="\t")
      print(top_tens.loc[topic_label+1,"Size"],end="\t")
      print(d_top_n_ngrams[topic_label][:3])
      result_txt.write("{}\t".format(topic_label))
      result_txt.write(top_tens.loc[topic_label+1,"Size"].__repr__()+"\t")
      for j in d_top_n_ngrams[topic_label][:3]:
        result_txt.write(j[0]+"\t")
      result_txt.write("\n")

#np.linalg.norm(umap_embeddings[702]-umap_embeddings[1734])
import nltk
import matplotlib.pyplot as plt

docs_df[docs_df["Topic"]==48]
nltk.FreqDist([i[3:10] for i in docs_df[docs_df["Topic"]==48]["Date"]])
nltk.FreqDist([i[3:10] for i in docs_df[docs_df["Topic"]==48]["Date"]]).most_common

fe = nltk.FreqDist([i[3:10] for i in docs_df[docs_df["Topic"]==48]["Date"]])
fe_dict = dict(fe)
temp_fe = fe_dict.items()
temp_fe = sorted(temp_fe) 
date_fe, count_fe = zip(*temp_fe) 

plt.plot(date_fe, count_fe)
plt.show()