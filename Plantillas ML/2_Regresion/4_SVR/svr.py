# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:27:30 2021

@author: HP
"""

#SVR
import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#ESCALADO DE VARIABLES
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#Creacion del modelo de regresion SVR
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X, y)

#Prediccion de los modelos SVR
y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

#invercion de la matriz

#Visualizacion del modelo SVR
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y)
plt.plot(X_grid, regression.predict(X_grid),color="green")
plt.title("Regresion SVR")
plt.xlabel("Nivel de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
