# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:14:45 2021

@author: HP
"""

#Regresion Lineal Multiple
import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   

#IMPORTAR EL DATASET
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#CODIFICAR DATOS CATEGORICOS
from sklearn import preprocessing
labelenconder_X = preprocessing.LabelEncoder()
X[:, 3] = labelenconder_X.fit_transform(X[:, 3])

        #VARIABLES DUMMY
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
        #TRANSFORMAMOS A VARIABLES DUMMY
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
        #COLOCAMOS LAS VARIABLES EN X   
X = np.array(ct.fit_transform(X), dtype=np.float)

pur = ColumnTransformer(
    [('one_hot_encoder', )])

#Evitar la trampa de las variables Dummy
X = X[:, 1:]

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Ajustar el modelo de regresion lineal multiple del conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de lo resultados en el conjunto de testing
y_pred = regression.predict(X_test)

#Construccion optimo (Eleminacion hacia atras)
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis=1)

#Automatico
def EliminacionAtras(x, sl):
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = X[:, [0,1,2,3,4,5]]
X_Model = EliminacionAtras(X_opt, SL)

#Uno por uno
'''
X_opt = X[:,[0,1,2,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:,[0,3,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:,[0,3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()
'''