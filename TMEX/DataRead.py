import json
import pandas

data_path = "./data/koreaherald_1517_"

data_files=[]

for corpus in range(8):
	with open( data_path + str(corpus) + ".json", 'r') as f:
		data=json.load(f)
	df = pandas.DataFrame.from_dict(data)
	data_files.append(df)
