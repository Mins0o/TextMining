print("Make sure you download\npip3 install sklearn hdbscan sentence_transformers umap_learn matplotlib nltk\npip3 install numpy --upgrade\n\n")
print("importing modules      ", end = "\r")
import json
import pandas as pd
import os
import pickle
import datetime

print("importing numpy    ", end = "\r")
import numpy as np

print("importing matplotlib.pyplot    ", end = "\r")
import matplotlib.pyplot as plt

print("importing nltk.FreqDist    ", end = "\r")
from nltk import FreqDist

print("importing vectorizer    ", end = "\r")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

print("importing sent_trans    ", end = "\r")
from sentence_transformers import SentenceTransformer

print("importing hdbscan        ",end = "\r")
import hdbscan

def c_tf_idf(documents, m, ngram_range=(4, 5)):
    vectorized = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    #vectorized = TfidfVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
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

def get_date(df, row_index):
  date = 0;
  try:
    date = datetime.datetime.strptime(df.loc[str(row_index)," time"][:10], "%Y-%m-%d")
  except:
    date = datetime.datetime.strptime(df.loc[row_index,"Date"][:10], "%Y-%m-%d")
  return date

def get_data(base_dir, data_dir, embeddings_file_name):
    data_path = base_dir + data_dir + "koreaherald_1517_"

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
        all_docs = pd.concat([all_docs,i.loc[:,[" body","title", " time"]]],sort = False)

    all_docs["uniqueID"] = list(range(len(all_docs)))

    y_2015 = all_docs[(all_docs[" time"]>"2015") & (all_docs[" time"]<"2016")]
    y_2016 = all_docs[(all_docs[" time"]>"2016") & (all_docs[" time"]<"2017")]
    y_2017 = all_docs[(all_docs[" time"]>"2017") & (all_docs[" time"]<"2019")]
    return([all_docs, y_2015, y_2016, y_2017])

def get_embeddings(data, embedding_file_path = "./checkpoints/embeddings"):
    # Doc2Vec
    if os.path.isfile(embedding_file_path+".mdl"):
        print(">> Using previous embeddings")
        with open(embedding_file_path+".mdl","rb") as embed_model:
            embeddings = pickle.load(embed_model)
    else:
        print(">> Generating new embeddings")
        model = SentenceTransformer('paraphrase-mpnet-base-v2')
        embeddings = model.encode(data, show_progress_bar=True)
        with open(embedding_file_path+".mdl","wb") as embed_model:
            pickle.dump(embeddings,embed_model)
    return(embeddings)

def get_reduction(embeddings, embedding_file_path = "./checkpoints/embeddings"):
    print("Reducing dimension")
    if os.path.isfile(embedding_file_path+"_umap"+".mdl"):
        print(">> Using previous reduction")
        with open(embedding_file_path+"_umap"+".mdl","rb") as umap_embeds:
            umap_embeddings = pickle.load(umap_embeds)
    else:
        print(">> Reducing embeddings again")
        import umap
        umap_embeddings = umap.UMAP(n_neighbors=20, n_components=7, min_dist = 0.02, metric='cosine').fit_transform(embeddings)  
        with open(embedding_file_path+"_umap"+".mdl","wb") as umap_embeds:
            pickle.dump(umap_embeddings,umap_embeds)
    return(umap_embeddings)

def cluster_selected_embeddings(embeddings, embedding_file_path = "./checkpoints/embeddings", selection_indices=None, selection_identifier_string="of all"):
    if (not selection_indices == None):
        embeddings = embeddings[selection_indices]

    print("Clustering")
    if os.path.isfile(embedding_file_path+"_"+"cluster"+"_"+selection_identifier_string+".mdl"):
        print(">> Using previous clusters")
        with open(embedding_file_path+"_"+"cluster"+"_"+selection_identifier_string+".mdl","rb") as cluster_file:
            cluster = pickle.load(cluster_file)
    else:
        print(">> Clustering embeddings again")
        cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(embeddings)
        with open(embedding_file_path+"_"+"cluster"+"_"+selection_identifier_string+".mdl","wb") as cluster_file:
            pickle.dump(cluster,cluster_file)
    return(cluster)

def rank_Ns_after_collect_by_column(text_output_path, topic_labeled_data, column_title="Title", N = 20, verbose = False):
    with open(text_output_path,'w') as result_txt:
        # Count how much is in each topics
        topic_sizes = extract_topic_sizes(topic_labeled_data)
        result_txt.write(str(len(topic_sizes))+"\n")
        top_Ns = topic_sizes.head(N)
        result_txt.write(top_Ns.to_string()+"\n")

        if(verbose):
            print("Number of Clusters: ",len(topic_sizes))
            print(top_Ns)

        concat_column_per_topic = topic_labeled_data.groupby(['Topic'], as_index = False).agg({column_title: ' '.join})
        #scoring ngrams in the collections
        print("scoring ngrams in the collections")
        tf_idf, count = c_tf_idf(concat_column_per_topic[column_title].values, m=len(topic_labeled_data))
        # Now all the documents are clustered
        # Extract_top_n_words_per_topic(tf_idf, count, per_topic, n=20)
        # From the headlines
        top_n_ngrams = extract_top_n_words_per_topic(tf_idf, count, concat_column_per_topic, n=N)

        # Print out the top_ten topics' top n ngrams
        for topic_label in top_Ns.loc[:,"Topic"]:
            result_txt.write("{}\t".format(topic_label))
            result_txt.write(top_n_ngrams[topic_label].__repr__())
            result_txt.write("\n")
            result_txt.write("\n")
        
        for topic_label in top_Ns.loc[:,"Topic"]:
            print(topic_label,end="\t")
            print(top_Ns.loc[topic_label+1,"Size"],end="\t")
            print(top_n_ngrams[topic_label][:2])
            result_txt.write("{}\t".format(topic_label))
            result_txt.write(top_Ns.loc[topic_label+1,"Size"].__repr__()+"\t")
            for j in top_n_ngrams[topic_label][:2]:
                result_txt.write(j[0]+"\t")
            result_txt.write("\n")
    return (top_Ns, top_n_ngrams)

def get_date_distribution_per_topic(topic_labeled_data, topic_number = -1):
    date_count = FreqDist([date_str[:10] for date_str in docs_df[docs_df["Topic"] == topic_number]["Date"]])
    dates, counts = zip(*sorted(dict(date_count).items()))
    dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
    return (dates, counts, date_count)

def get_centroids_and_variance(text_output_path, topic_labeled_data, embeddings, topics=[-1], verbose = False):
    with open(text_output_path,'a') as result_txt:
        result_txt.write("\n")
        topic_centroids = []
        topic_variances = []
        for selected_topic_i in topics:
            indices_per_topic = topic_labeled_data[topic_labeled_data["Topic"] == selected_topic_i]["Doc_ID"]
            centroid = np.mean(embeddings[indices_per_topic], axis = 0)
            variance = np.mean(np.sqrt(np.sum((embeddings[indices_per_topic] - centroid) ** 2, axis = 1)))
            topic_centroids.append(centroid)
            topic_variances.append(variance)
            if(verbose):
                print(selected_topic_i, "\t", np.array2string(centroid,precision = 4, separator = ", "), "\t", variance)
            result_txt.write(str(selected_topic_i) + "\t" + np.array2string(centroid, precision = 4, separator = "\t") + "\t" + str(variance) +"\n")
        return(topic_centroids, topic_variances)

base_dir = "/Volumes/Personal files/Workspace/GitHub/TextMining"
data_dir = "/data/"

result_dir = base_dir+"/results/"
t_result_file_name = result_dir + "Titling_t"
d_result_file_name = result_dir + "Titling_d"
embedding_file_name = "functionized"

# The embeddings of the documents can be saved with pickle
check_points_dir = base_dir+"/checkpoints/"

umap_embd_file_name = check_points_dir+"umap_"+embedding_file_name
cluster_file_name = check_points_dir+"cluster_"+embedding_file_name
embedding_file_name = check_points_dir+embedding_file_name

plot_graphs = False
title_extraction = True
body_extraction = True
topic_space_info = True

targets = get_data(base_dir, data_dir, embedding_file_name)
target_names = ["all_docs", "2015","2016","2017"]

all_docs = targets[0]


for target_num in range(4):
    print("\n\n---------------------------------",target_names[target_num],"----------------------------\n\n")
    target_collection = targets[target_num]
    print(len(target_collection)," articles")
    indices = list(target_collection.loc[:,"uniqueID"])

    embeddings = get_embeddings(all_docs.loc[:," body"],embedding_file_name)
    reduced_embeddings = get_reduction(embeddings, embedding_file_name)
    cluster = cluster_selected_embeddings(reduced_embeddings, embedding_file_name, indices, target_names[target_num])

    # constructing a data frame:
    # Rows  |Doc(text body)  |Doc_ID |Topic(labels)  |Title(headline) |Date
    #       |                |       |               |                |
    docs_df = pd.DataFrame(list(target_collection.loc[:," body"]), columns=["Doc"])
    docs_df['Doc_ID'] = list(target_collection.loc[:,"uniqueID"])
    docs_df['Topic'] = cluster.labels_
    docs_df["Title"] = list(target_collection.loc[:,"title"])
    docs_df["Date"] = list(target_collection.loc[:," time"])


    if(body_extraction and target_num): 
        print("\n\nFrom Text Bodies\n\n")
        body_output_path = d_result_file_name+"_"+target_names[target_num]+".txt"
        rank_Ns_after_collect_by_column(text_output_path = body_output_path, topic_labeled_data = docs_df, column_title = "Doc")

    if(title_extraction):
        print("\n\nFrom Headlines\n\n")
        title_output_path = t_result_file_name+"_"+target_names[target_num]+".txt"
        top_Ns, top_N_ngrmas = rank_Ns_after_collect_by_column(title_output_path, docs_df, column_title = "Title",N=20)

        #np.linalg.norm(umap_embeddings[702]-umap_embeddings[1734])

    if (plot_graphs and title_extraction):
        dates, counts, date_count = get_date_distribution_per_topic(docs_df, top_Ns["Topic"].values[1])
        print(date_count)
        plt.plot(dates,counts)

    if (title_extraction and topic_space_info):
        top_topics = top_Ns["Topic"].values
        get_centroids_and_variance(title_output_path, docs_df, reduced_embeddings, top_topics, True)
        print()
        whole_centroid = np.mean(reduced_embeddings[docs_df["Doc_ID"]])
        whole_variance = np.mean(np.sqrt(np.sum((reduced_embeddings[docs_df["Doc_ID"]] - whole_centroid)**2,axis =1)))
        print(whole_centroid,"\t",whole_variance)
        

if (plot_graphs and title_extraction):
    plt.show()
        