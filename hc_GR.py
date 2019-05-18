#####################################################################################
# Creator     : Gaurav Roy
# Date        : 18 May 2019
# Description : The code performs Hierarchical Clustering on the Mall_Customers.csv.
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

# Using Dendogram to find optimal number of clusters
'''
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.grid(True)
plt.show()
# The largest distance was on the right most line after the first split.
# Drawing the threshold line through that, we get : 5 clusters
'''

# Applying Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
Y_hc = hc.fit_predict(X)

# Visualize the clusters
# Plotting with generic cluster labels
'''
# Plotting the datapoints according to the clusters they belong in
plt.scatter(X[Y_hc==0, 0], X[Y_hc==0, 1], s=100, c='red', label='Cluster 1', edgecolors='black')
plt.scatter(X[Y_hc==1, 0], X[Y_hc==1, 1], s=100, c='blue', label='Cluster 2', edgecolors='black')
plt.scatter(X[Y_hc==2, 0], X[Y_hc==2, 1], s=100, c='green', label='Cluster 3', edgecolors='black')
plt.scatter(X[Y_hc==3, 0], X[Y_hc==3, 1], s=100, c='cyan', label='Cluster 4', edgecolors='black')
plt.scatter(X[Y_hc==4, 0], X[Y_hc==4, 1], s=100, c='magenta', label='Cluster 5', edgecolors='black')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
'''


# Renaming the cluster names based on their properties

# Plotting the datapoints according to the clusters they belong in

# Careful clients: LOW Spending score but HIGH Salary
plt.scatter(X[Y_hc==0, 0], X[Y_hc==0, 1], s=100, c='red', label='Careful', edgecolors='black')
# Standard clients: AVERAGE Spending score and AVERAGE Salary
plt.scatter(X[Y_hc==1, 0], X[Y_hc==1, 1], s=100, c='blue', label='Standard', edgecolors='black')
# Target clients: HIGH Spending score and HIGH salary
plt.scatter(X[Y_hc==2, 0], X[Y_hc==2, 1], s=100, c='green', label='Target', edgecolors='black')
# Careless clients: HIGH Spending score but LOW Salary
plt.scatter(X[Y_hc==3, 0], X[Y_hc==3, 1], s=100, c='cyan', label='Careless', edgecolors='black')
# Sensible clients: LOW Spending score and LOW Salary
plt.scatter(X[Y_hc==4, 0], X[Y_hc==4, 1], s=100, c='magenta', label='Sensible', edgecolors='black')

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()