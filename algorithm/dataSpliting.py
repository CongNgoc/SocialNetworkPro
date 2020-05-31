import pandas as dp



# Open file from resources 
file_questions = '../resources/result_prep_questions.csv'
file_answers = '../resources/result_prep_answers.csv'

df_questions = dp.read_csv(file_questions)
df_answers = dp.read_csv(file_answers)

print(df_questions)
# Dataset splitting

# X, y = np.arange(10).reshape((5, 2)), range(5)
#     X_train, X_test, y_train, y_test = train_test_split(data_tokenization_lower, data_tokenization_lower, train_size=0.7, random_state=42)

#     # # Print Data
#     print(X)
# print(y)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
# # print(tokens_without_sw)


