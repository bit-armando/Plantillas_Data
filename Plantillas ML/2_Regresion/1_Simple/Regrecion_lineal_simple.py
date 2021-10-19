# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:57:44 2021

@author: HP
"""

#Regresion lineal simple
import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#ESCALADO DE VARIABLES
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Crear modelo de regresion lineal simple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo")
plt.show()

#Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "green")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de test)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo")
plt.show()