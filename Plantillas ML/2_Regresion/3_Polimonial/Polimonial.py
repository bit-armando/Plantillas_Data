#REGRESION POLINOMICA
import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(X,y)

#Ajustar la regresion pilinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
Pol_reg = PolynomialFeatures(degree = 5)
X_pol = Pol_reg.fit_transform(X)

Lin_reg2 = LinearRegression()
Lin_reg2.fit(X_pol,y)

#Visualizacion del modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, Lin_reg.predict(X),color="blue")
plt.title("Regresion Lineal")
plt.xlabel("Nivel de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

#Visualizacion del modelo Polinomico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y)
plt.plot(X_grid, Lin_reg2.predict(Pol_reg.fit_transform(X_grid)),color="green")
plt.title("Regresion Polinomica")
plt.xlabel("Nivel de empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

#Prediccion de los modelos
Lin_reg.predict([[6.5]])
Lin_reg2.predict(Pol_reg.fit_transform([[6.5]]))


