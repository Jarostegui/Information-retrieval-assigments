import pandas as pd
import numpy as np
np.seterr(divide = 'ignore') 

#Original
#consulta = [0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0]

#Nueva
consulta = [0,0,0,0,0,0.365,0,0,0,0,0,0,0.071,0,0,0,0,0,1.821,-0.003]


data = pd.read_csv('terminos.txt')

df_tf_idf = data.copy().drop(columns='url').set_index('documento')

tf = 1 + np.log2(df_tf_idf)
tf = tf.replace(np.NINF, 0)

df_tf_idf[df_tf_idf > 1] = 1
idf = df_tf_idf.sum()
idf = np.log(21/(idf+0.5))

tf_idf = tf * idf

dq = (consulta*tf_idf).sum(axis=1)
d = np.sqrt((tf_idf**2).sum(axis=1))

rank1 = (dq / d).sort_values(ascending=False)
print(rank1)
