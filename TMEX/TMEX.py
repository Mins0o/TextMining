import nltk.tokenize
from nltk.corpus import stopwords
import json
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
import DataRead

data_files = DataRead.data_files

import TrendAnalysis01
#import Sections
import LDA_exp

for i in data_files:
    for j in range(10):
        print(i.loc[str(j+1),"title"])
        print(i.loc[str(j+1)," body"])

input("\nend")