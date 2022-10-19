import pandas as pd
import numpy as np
np.seterr(divide = 'ignore') 

alfa = 1.0
beta = 1.0
gamma = 1.0
D = 1000
r = 8
cutoff = 4

consulta = [0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0]

data = pd.read_csv('terminos.txt')

df_tf_idf = data.copy().drop(columns='url').set_index('documento')

tf = 1 + np.log2(df_tf_idf)
tf = tf.replace(np.NINF, 0)

df_tf_idf[df_tf_idf > 1] = 1
idf = df_tf_idf.sum()
idf = np.log(21/(idf+0.5))

tf_idf = tf * idf

relevancias = pd.read_csv('database-google.txt')
relevancias = relevancias[['url', 'relevancia']].set_index('url')

database = pd.merge(data[['url', 'documento']], relevancias, on=['url']).drop(columns=['url'])
database = database.set_index('documento')
database[database > 1] = 1
Dr = float(database.sum(0))
Dn = float(database.count(0) - Dr)

database = tf_idf.merge(database, on=['documento'])
rels = database[database['relevancia'] > 0]
rels = rels.sum(0).drop('relevancia')
rels = rels / Dr

norels = database[database['relevancia'] == 0]
norels = norels.sum(0).drop('relevancia')
norels = norels / Dn

consulta = pd.Series(consulta, index=rels.index)

q = alfa * consulta + beta*rels - gamma*norels

print(q)

Dw = database.astype(bool).sum(0).drop('relevancia')
rw = database[database['relevancia'] > 0 ].astype(bool).sum(0).drop('relevancia')

bir = np.log((rw + 0.5)*(D - Dw - r + rw + 0.5) / (Dw - rw +0.5)*(r - rw + 0.5))

rsv = rw * bir
rsv = rsv.sort_values(ascending=False)

print(rsv.iloc[:cutoff])
