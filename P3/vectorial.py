import pandas as pd
import numpy as np
np.seterr(divide = 'ignore') 

data = pd.read_csv('datos.txt')
data = data.set_index('documento')

tf = 1 + np.log2(data)
tf = tf.replace(np.NINF, 0)

database = data.copy()
database[database > 1] = 1
idf = database.sum()
idf = np.log(21/(idf+0.5))

tf_idf = tf * idf

consulta = [0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0]

dq = (consulta*tf_idf).sum(axis=1)
d = np.sqrt((tf_idf**2).sum(axis=1))

rank1 = (dq / d).sort_values(ascending=False)
print(rank1)
