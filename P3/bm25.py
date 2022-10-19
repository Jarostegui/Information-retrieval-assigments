import pandas as pd
import numpy as np
np.seterr(divide = 'ignore') 

def rsj_funct(data, consulta):
    #numDocs = data.count(axis=1)[0]
    numDocs = 1000
    a = (consulta*data)
    numDocsWord = a[a>1]
    numDocsWord = numDocsWord.count()
    numDocsWord = numDocsWord[numDocsWord != 0]
    result = np.log((numDocs - numDocsWord + 0.5) / (numDocsWord + 0.5))
    return result

data = pd.read_csv('datos.txt')
data = data.set_index('documento')
consulta = [0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0]

rsj = rsj_funct(data, consulta)

longDoc = data.sum(axis=1)
avgLong = np.mean(longDoc)
normLong = longDoc/avgLong

numerador = data.mul(rsj, axis =1)
denominador = data.add(normLong, axis = 0)

result = numerador.div(denominador, axis = 0)
print(result.sum(axis = 1).sort_values(ascending=False))