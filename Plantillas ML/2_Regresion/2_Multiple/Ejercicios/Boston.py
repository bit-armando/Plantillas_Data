import numpy as np                
from sklearn import datasets

boston = datasets.load_boston()
x = boston.data[:,:]
y = boston.target

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Ajustar el modelo de regresion lineal multiple del conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de lo resultados en el conjunto de testing
y_pred = regression.predict(X_test)

#Construccion optimo (Eleminacion hacia atras)
import statsmodels.api as sm

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
X_opt = x[:, :]
X_Model = EliminacionAtras(X_opt, SL)

print(regression.score(X_test,y_test))