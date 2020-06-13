from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans, MiniBatchKMeans

import pandas as dp

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import sys
import matplotlib.pyplot as plt

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# DATA PRE-PROCESSING
# Open file from resources 
file_questions = '../resources/QueryResults4Questions.csv'
df_questions = dp.read_csv(file_questions)
# questions_train_set = df_question
df_question, questions_test_set = train_test_split(df_questions, train_size=0.7, random_state=42)

# RUN TfidfVectorizer
print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5,
                             min_df=2, stop_words='english')
X = vectorizer.fit_transform(df_question['Body'])
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

# APPLY ELBOW
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

print(distortions)    

# Plotting the distortions of K-Means
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# APPLY KMEANS TO FIND CLUSTERS
true_k = 4
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
