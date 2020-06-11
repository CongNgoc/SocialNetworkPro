from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

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

# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]




# Open file from resources 
file_questions = '../resources/QueryResults4Questions.csv'

df_question = dp.read_csv(file_questions)
# print(df_question['Body'])
# sys.exit()

# DATASET TYPE LA BUNCH
# DATASET.DATA TYPE LA LIST
# DATASET.DATA ELEMENT TYPE LA STRING


# Uncomment the following to do the analysis on all the categories
# categories = None

# print("Loading 20 newsgroups dataset for categories:")
# print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)


# CHECK INFO DATASET
# index_check = 0
# print("size: " + str(len(dataset.data)))
# print("type: " + str(type(dataset.data)))

# for i_dataset in dataset.data:
#     print(i_dataset)
#     print("type: " + str(type(i_dataset)))
#     if index_check == 1:
#         sys.exit() 

#     index_check = index_check + 1


# print("%d documents" % len(dataset.data))
# print("%d categories" % len(dataset.target_names))
# print("%d documents" % len(dataset.target))
# print("%d categories" % len(dataset.target_names))

labels = dataset.target
true_k = np.unique(labels).shape[0]
# NOI VAO DAY


# FIND k use Elbow
# Runing KMeans with a range of k

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()


print("====================RUN TfidfVectorizer HERE!")
vectorizer = TfidfVectorizer(max_df=0.5,
                             min_df=2, stop_words='english')

# X = vectorizer.fit_transform(dataset.data)
X = vectorizer.fit_transform(df_question['Body'])

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()


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



# END FIND k use Elbow

# BAT DAU LAI O DAY
true_k = 6

# print("====================true_k" + str(true_k))

# #############################################################################
# Do the actual clustering

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

# print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)


# print("done in %0.3fs" % (time() - t0))
# print()

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))

# print()



print("Top terms per cluster:")

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
