# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 19:26:18 2021

@author: Armando Rosales
"""

#Clustering jerarquico
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importar los datos con pandas
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

#Utilizar el dendograma
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidia")
plt.show()

#Ajustar el clustering al conjunto X
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean")
y_hc = hc.fit_predict(X)

#Graficamos el modelo
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s=100, color="red",label="Cluster1")
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s=100, color="blue",label="Cluster2")
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s=100, color="green",label="Cluster3")
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s=100, color="cyan",label="Cluster4")
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s=100, color="magenta",label="Cluster5")

plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()