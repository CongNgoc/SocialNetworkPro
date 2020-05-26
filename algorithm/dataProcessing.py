import pandas as dp
from nltk.tokenize import word_tokenize
import re


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

file_name = '../resources/QueryResults4Answer.csv'
data_frame = dp.read_csv(file_name)

#
# print(data_frame['Body'][0])
clean_text = cleanhtml(data_frame['Body'][0])

# Data Tokenization
data_tokenization = word_tokenize(clean_text)

# Lower case conversion 
data_tokenization_lower = [x.lower() for x in data_tokenization]
print(data_tokenization_lower)









