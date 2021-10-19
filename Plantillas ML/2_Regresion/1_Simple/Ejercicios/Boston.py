import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


#Importasmos el dataset y asignamos x para cada y
boston = datasets.load_boston()
x = boston.data[:, np.newaxis, 5]
y = boston.target

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Crear modelo de regresion lineal simple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Costo de vivienda de acuerdo habitacion (Conjunto de entrenamiento)")
plt.xlabel("Numero habitaciones")
plt.ylabel("Costo vivienda")
plt.grid(True)
plt.minorticks_on()
plt.show()

#Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "green")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Costo de vivienda de acuerdo habitacion (Conjunto de test)")
plt.xlabel("Numero habitaciones")
plt.ylabel("Costo vivienda")
plt.grid(True)
plt.minorticks_on()
plt.show()

print("Presicion del modelo: ")
print(regression.score(X_train,y_train))