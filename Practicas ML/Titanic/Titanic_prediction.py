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
data_test = pd.read_csv('test.csv')

y_train = data_train.iloc[:,1].values
X_train = data_train.iloc[:, [2,4,5,6,7,11]].values

X_test = data_test.iloc[:, [1,3,4,5,6,10]].values
y_test = pd.read_csv('gender_submission.csv')
y_test = y_test.iloc[:,1].values


#VISUALIZACION DE LOS DATOS
data_train.groupby('Sex')['Survived'].sum().plot(kind='bar', legend=True, rot=0)
plt.title("SOBREVIVIENTES POR SEXO")
plt.show()

data_train.groupby('Embarked')['Survived'].sum().plot(kind='bar',legend=True, rot=0)
plt.title('SOBREVIVIENTES POR LUGAR DE EMBARCAMIENTO')
plt.show()

data_train.groupby('Pclass')['Survived'].sum().plot(kind='bar',legend=True, rot=0)
plt.title('SOBREVIVIENTES POR CLASE A LA QUE PERTENECE')
plt.show()

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

#CONSTRUCCION DEL MODELO
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=500, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#MATRIZ DE CONFUCION
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,y_pred)