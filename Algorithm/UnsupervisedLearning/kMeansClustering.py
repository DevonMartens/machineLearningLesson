import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

 # generate random data into a 2d space it will make two 
X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = ‘b’)
plt.show()

# process data data
from sklearn.cluster import KMeans
# we  gave k (n_clusters) an arbitrary value of two
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

#find the center of the clusters:
Kmean.cluster_centers_

#display the cluster centroids (using green and red color)
plt.scatter(X[ : , 0], X[ : , 1], s =50, c=’b’)
plt.scatter(-0.94665068, -0.97138368, s=200, c=’g’, marker=’s’)
plt.scatter(2.01559419, 2.02597093, s=200, c=’r’, marker=’s’)
plt.show()

#show algorithm for how data points are categorized
Kmean.labels_

#predict the cluster of a data point it
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)
