import pandas as dp
from sklearn.model_selection import train_test_split


# Open file from resources 
file_questions = '../resources/result_prep_questions.csv'
file_answers = '../resources/result_prep_answers.csv'

df_questions = dp.read_csv(file_questions)
df_answers = dp.read_csv(file_answers)

# Dataset splitting
questions_train_set, questions_test_set = train_test_split(df_questions, train_size=0.7, random_state=42)
ansers_train_set, ansers_test_set = train_test_split(df_answers, train_size=0.5, random_state=42)
# # Print Data
# print(X)
# print(y)
print(questions_train_set['Id'][[5191]])

# Get item from dataframe
print(type(questions_train_set))
print("=========================================================")
# print(ansers_train_set)
# print(X_test)
# print(y_train)
# print(y_test)
# print(tokens_without_sw)


