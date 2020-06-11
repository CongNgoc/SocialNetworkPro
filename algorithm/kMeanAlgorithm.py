import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
from sklearn.cluster import KMeans
from sklearn import datasets
from dataSpliting import splitDataset


# DATASET
iris = datasets.load_iris()
# questions_train_set, questions_test_set, ansers_train_set, ansers_test_set = splitDataset()

#we are usingh
# df=pd.DataFrame(questions_train_set)
df=pd.DataFrame(iris['data'])
# print(df)


# Runing KMeans with a range of k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(iris['data'])
    distortions.append(kmeanModel.inertia_)
    print(kmeanModel.inertia_)

print(distortions)    
print("=======Gia tri cua K %d" %K)

# Plotting the distortions of K-Means

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
# plt.show()


kmeanModel = KMeans(n_clusters=3)
kmeanModel.fit(iris['data'])

df['k_means']=kmeanModel.predict(iris['data'])
df['target']=iris['target']
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(df[0], df[1], c=df['target'])
axes[1].scatter(df[0], df[1], c=df['k_means'], cmap=plt.cm.Set1)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('K_Means', fontsize=18)
# plt.show()