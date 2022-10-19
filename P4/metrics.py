import pandas as pd
import numpy as np
p = 0.8
D = 20

def dcg_funct(df):
    return np.sum(df.loc[:]['relevancia'] / np.log2(df.loc[:]['posicion'] + 1))

def idcg_funct(data):
    result = 0.0
    for index, value in enumerate(data):
        result += value/np.log2(index+2)
    return result


def calcularMetricas(consulta):
    print("\n#####"+consulta+"#####")

    df = pd.read_csv(consulta)

    #Precision@3
    print("Precision@3: "+str(df.head(3)['relevancia'][df['relevancia'] > 0].count()/3))
    #Precision@5
    print("Precision@5: "+str(df.head(5)['relevancia'][df['relevancia'] > 0].count()/5))
    #Precision@10
    print("Precision@10: "+str(df.head(10)['relevancia'][df['relevancia'] > 0].count()/10))

    #nDCG
    dcg = dcg_funct(df)
    poolActual = df[df['relevancia'] > 0]
    
    poolActual = poolActual.sort_values(by='relevancia', ascending=False)
    idcg = idcg_funct(poolActual['relevancia'])
    print("nDCG: "+ str(dcg/ idcg))


calcularMetricas("consultaOriginal.txt")
calcularMetricas("consultaModificada.txt")
calcularMetricas("consultaOriginalGoogle.txt")
calcularMetricas("consultaModificadaGoogle.txt")
