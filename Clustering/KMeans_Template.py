#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:13:32 2018

@author: kusal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#---------------------Importing the Datasets------------------------------------------
dataset = pd.read_csv("preprocessed_trips_discretised.csv")
X = dataset.iloc[:,[17,21]].values       # Columns of interest 17- Distance, 21- Duration

#------Using the elbow method to find the optimal number of clusters------------------
within_cluster_sum_of_squares = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, n_jobs=-1,
                    verbose=5, max_iter=300, random_state = 0)
    kmeans.fit(X)
    within_cluster_sum_of_squares.append(kmeans.inertia_)

#Plotting elbow method graph
plt.plot(range(1,11), within_cluster_sum_of_squares)
plt.title("Elbow method computation")
plt.xlabel("No of clusters")
plt.ylabel("wcss")
plt.show()

#---------------------------------Applying Kmeans to Dataset-------------------------
kmeans = KMeans(n_clusters=5, n_jobs=-1)
y_kmeans = kmeans.fit_predict(X)

#------------------------------Visualizing the Results-------------------------------
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], c='r', label='cluster_1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], c='b', label='cluster_2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], c='g', label='cluster_3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c='c', label='cluster_4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], c='m', label='cluster_4')

#Plotting Centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            c='y', label="Centroids")
plt.title("Distance vs Duration Clusters")
plt.xlabel("Distance")
plt.ylabel("Duration")
plt.legend()
plt.show()
