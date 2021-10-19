# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:32:08 2021

@author: HP
"""
#Bosques Aleatorios
import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Ajustar el random forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regresor = RandomForestRegressor(n_estimators=10 ,random_state=0)
regresor.fit(X, y)

#Prediccion del modelo
y_pred = regresor.predict([[6.5]])

#Visualizacion del modelo Polinomico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y)
plt.plot(X_grid, regresor.predict(X_grid),color="green")
plt.title("Regresion Polinomica")
plt.xlabel("Nivel de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

