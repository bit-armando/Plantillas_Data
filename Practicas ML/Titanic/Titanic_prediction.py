# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:16:16 2021

@author: Armando Rosales
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#SEPARACION DE LOS DATOS
data_train = pd.read_csv('train.csv')
y_train = data_train.iloc[:,1].values
X_train = data_train.iloc[:, [2,4,5,6,7,11]].values

data_test = pd.read_csv('test.csv')
X_test = data_test.iloc[:, [1,3,4,5,6,10]].values
y_test = pd.read_csv('gender_submission.csv')
y_test = y_test.iloc[:,1].values


#TRATAMIENTO DE LOS  NAs
from sklearn.impute import SimpleImputer

def NAsData(frame, estrategia):
    imputer = SimpleImputer(missing_values=np.nan, strategy=estrategia, fill_value=None, verbose=0, copy=True)
    imputer = imputer.fit(frame)
    frame=imputer.transform(frame)
    return frame
    

X_train[:,2:3]=NAsData(X_train[:,2:3],'mean')
X_train[:,5:6]=NAsData(X_train[:,5:6],'most_frequent')

X_test[:,2:3]=NAsData(X_test[:,2:3],'mean')
X_test[:,5:6]=NAsData(X_test[:,5:6],'most_frequent')

#CODIFICAR DATOS CATEGORICOS
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

labelenconder_X = preprocessing.LabelEncoder()
X_train[:, 1] = labelenconder_X.fit_transform(X_train[:, 1])
X_test[:, 1] = labelenconder_X.fit_transform(X_test[:, 1])

labelenconder_X = preprocessing.LabelEncoder()
X_train[:, 5] = labelenconder_X.fit_transform(X_train[:, 5])
X_test[:, 5] = labelenconder_X.fit_transform(X_test[:, 5])

        #TRANSFORMAMOS A VARIABLES DUMMY
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [5])],   
    remainder='passthrough'                        
)
        #COLOCAMOS LAS VARIABLES EN X   
X_train = np.array(ct.fit_transform(X_train), dtype=np.float)
X_train=np.delete(X_train,2,axis=1)
X_test = np.array(ct.fit_transform(X_test), dtype=np.float)
X_test=np.delete(X_test,2,axis=1)

'''
VISUALIZACION DE X_train y X_test
0 y 1: Embarked
2    : Pclass
3    : Sex
4    : Age
5    : SibSp
6    : Parch
'''

#VISUALIZACION DE LOS DATOS
X_edad = np.array(["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80"])
X_genero = np.array(["male","female"])


y_edad = np.zeros(8)
y_genero = np.zeros(2)

def sumaCaracteristicas(arrayX, arrayY, indiceX, indiceY, cond1, cond2, i):
    if arrayX[i,indiceX]>=cond1 and arrayX[i,indiceX]<=cond2:
            arrayY[indiceY] = arrayY[indiceY]+1
    return arrayY[indiceY]        

for i in range(0,891):
    if y_train[i]==1:
        #SUMA POR EDADES
        y_edad[0] = sumaCaracteristicas(X_train,y_edad,4, 0, 0, 10, i)
        y_edad[1] = sumaCaracteristicas(X_train,y_edad,4, 1, 11, 20, i)
        y_edad[2] = sumaCaracteristicas(X_train,y_edad,4, 2, 21, 30, i)
        y_edad[3] = sumaCaracteristicas(X_train,y_edad,4, 3, 31, 40, i)
        y_edad[4] = sumaCaracteristicas(X_train,y_edad,4, 4, 41, 50, i)
        y_edad[5] = sumaCaracteristicas(X_train,y_edad,4, 5, 51, 60, i)
        y_edad[6] = sumaCaracteristicas(X_train,y_edad,4, 6, 61, 70, i)
        y_edad[7] = sumaCaracteristicas(X_train,y_edad,4, 7, 71, 80, i)
        
        #SUMA POR GENERO
        if X_train[i,3]==1:
            y_genero[0] = y_genero[0]+1
        else:
            y_genero[1] = y_genero[1]+1
            

plt.barh(X_edad, y_edad)
plt.ylabel("Edad pasajero")
plt.xlabel("Cantidad Sobrevivientes")
plt.title("Edad de Sobrevivientes")
plt.show()

plt.bar(X_genero, y_genero)
plt.xlabel("Genero pasajero")
plt.ylabel("Cantidad Sobrevivientes")
plt.title("Genero de Sobrevivientes")
plt.show()


#CONSTRUCCION DEL MODELO
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=500, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#MATRIZ DE CONFUCION
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,y_pred)