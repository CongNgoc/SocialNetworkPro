import pandas as dp
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
from htmllaundry import strip_markup


# Open file from resources 
file_questions = '../resources/QueryResults4Questions.csv'
file_answers = '../resources/QueryResults4Answers.csv'

def preProcessingData(file_name):
    data_frame = dp.read_csv(file_name)

    # Data Tokenization
    list_tokenization = [strip_markup(clean_text).split(" ") for clean_text in data_frame['Body']]

    # Lower case conversion and Removal of Stop words
    data_tokenization_lower = []
    for tokenization  in list_tokenization:
        tokenization_lower = []
        for x in tokenization:
            x_lower     = x.lower().replace('\n', ' ')
            if x_lower not in stopwords.words('english'):  
                tokenization_lower.append(x_lower)

        data_tokenization_lower.append(tokenization_lower)

    data_frame['Body'] = data_tokenization_lower
    return data_frame

df_of_questions = preProcessingData(file_questions)
df_of_answers   = preProcessingData(file_answers)

df_of_questions.to_csv('../resources/result_prep_questions.csv', index = False)
df_of_answers.to_csv('../resources/result_prep_answers.csv', index = False)

print('dataProcessing is done!')







