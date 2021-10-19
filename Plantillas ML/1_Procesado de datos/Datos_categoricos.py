# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:10:06 2021

@author: Armando Rosales
"""

#COMO IMPORTAR LIBRERIAS
import numpy as np                #LIBRERIA COMPLETA
import pandas as pd
import matplotlib.pyplot as plt   #SUB LIBRERIA

#IMPORTAR EL DATASET
dataset = pd.read_csv('Data.csv') #Carga del data set

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#CODIFICAR DATOS CATEGORICOS
from sklearn import preprocessing
labelenconder_X = preprocessing.LabelEncoder()
X[:, 0] = labelenconder_X.fit_transform(X[:, 0])

        #VARIABLES DUMMY
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
        #TRANSFORMAMOS A VARIABLES DUMMY
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
        #COLOCAMOS LAS VARIABLES EN X   
X = np.array(ct.fit_transform(X), dtype=np.float)

pur = ColumnTransformer(
    [('one_hot_encoder', )])

labelenconder_y = preprocessing.LabelEncoder()
y = labelenconder_y.fit_transform(y)