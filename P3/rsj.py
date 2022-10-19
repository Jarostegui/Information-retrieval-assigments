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
data[data > 1] = 1
rsj = rsj*data
result = rsj.sum(axis=1).sort_values(ascending=False)

print(result)

