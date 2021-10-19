# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:49:50 2021

@author: HP
"""

import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#ESCALADO DE VARIABLES
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


#Ajustar la regresion con el dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Prediccion de los modelos
y_pred = regression.predict(([[6.5]])) 


#Visualizacion del modelo Polinomico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y)
plt.plot(X_grid, regression.predict((X_grid)),color="green")
plt.title("Regresion")
plt.xlabel("Nivel de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
