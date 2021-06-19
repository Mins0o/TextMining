import DataRead
import pandas

data_files = DataRead.data_files

data=[]
sections =[]
for i in data_files:
    print(type(i))
    sections += list(i[" section"])

print(set(sections))

