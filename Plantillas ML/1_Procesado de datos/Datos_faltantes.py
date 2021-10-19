# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:12:43 2021

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

#TRATAMIENTO DE LOS  NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])