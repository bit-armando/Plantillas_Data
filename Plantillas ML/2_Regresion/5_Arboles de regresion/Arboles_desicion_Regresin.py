# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:27:42 2021

@author: HP
"""
#Regresion con arboles de desicion
import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
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
"""

#Ajustar la regresion con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X,y)

#Prediccion del modelo
y_pred = regression.predict([[6.5]])

#Visualizacion del modelo SVR
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y)
plt.plot(X, regression.predict(X),color="green")
plt.title("Regresion SVR")
plt.xlabel("Nivel de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()