#Natural Language Processing (NLP)
import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

#Limpieza de texto
import re
import nltk as nl
#Lo que se descarga son las palabras que no aportan mucho a una valoracion 
nl.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

'''
Quitamos todo aquel caracter que no sea una letra de la a-z
luego lo campiamos a minusculas y lo separamos por palabras
Despues de eso pasamos al proceso de eliminacion de palabras inecesarias tanto como
  tambien pasamos a la raiz de un verbo 
'''
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Crear el Bag of Words
    #De palabra a vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#------------------Algoritmo de Bosque aleatorio(Clasificacion)----------------------------------
    #DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Ajustar el Clasificador con el conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Prediccion de los resultados
y_pred = classifier.predict(X_test)

#Matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,y_pred)
