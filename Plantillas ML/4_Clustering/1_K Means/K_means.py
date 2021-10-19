# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:59:15 2021

@author: Armando Rosales
"""
#K-means
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Metodo del codo para averiguar el num de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300,n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

#Aplicar el metodo de K-means para segmentar el dataset
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300,n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizar clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s=100, color="red",label="Tacaños")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s=100, color="blue",label="Estandar")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s=100, color="green",label="Objetivo")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s=100, color="cyan",label="Descuidados")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s=100, color="magenta",label="Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, color="yellow",label="Baricentros")

plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()