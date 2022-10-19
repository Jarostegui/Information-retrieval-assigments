import pandas as pd
import numpy as np
p = 0.8

def dcg_funct(df):
    return np.sum(df.loc[:]['relevancia'] / np.log2(df.loc[:]['posicion'] + 1))

def idcg_funct(data):
    result = 0.0
    for index, value in enumerate(data):
        result += value/np.log2(index+2)
    return result

def rbp_funct(df):
    result = 0.0
    for index, row in df.iterrows():
        exp = row['posicion'] - 1
        result += p**exp * row['relevancia']
    return result

def err_funct(df):
    result = 0.0
    proba = 1.0
    for index, row in df.iterrows():
        k = row['posicion']
        utility = (2**row['relevancia'] - 1) / 4
        result += proba * utility / k
        proba *= (1-utility)
    return result


def calcularMetricas(buscador):
    print("\n#####"+buscador+"#####")

    df = pd.read_csv(f'datosOffline/database-{buscador}.txt')

    precision = []
    recall = []
    map = []
    mrr = []
    ndcg = []
    err = []
    rbp = []

    for consul in df['consulta'].unique():
        consulActual = df.loc[df['consulta'] == consul]

        #Precision
        prec = consulActual['relevancia'][consulActual['relevancia'] > 0].count() / len(consulActual)
        precision.append(prec)

        #Recall
        qrels = pd.read_csv(f'datosOffline/qrels.txt')

        poolActual = qrels.loc[qrels['consulta'] == consul]
        poolActual = poolActual[poolActual['relevancia'] > 0]

        rec = consulActual['relevancia'][consulActual['relevancia'] > 0].count() / len(poolActual)
        recall.append(rec)

        #MAP
        avgMap = []
        for index, row in consulActual.iterrows():
            if row['relevancia'] > 0:
                pAtRow = 0
                for indexe, c in consulActual.iterrows():
                    if  c['posicion'] < row['posicion']:
                        pAtRow += c['relevancia']
                pAtRow /= row['posicion']
                avgMap.append(pAtRow)
        if not avgMap:
            avgMap = [0]
        map.append(np.mean(avgMap))

        #MRR
        for index, row in consulActual.iterrows():
            if row['relevancia'] > 0:
                mrr.append(1/row['posicion'])
                break

        #ndcg   
        dcg = dcg_funct(consulActual)
        poolActual = poolActual.sort_values(by='relevancia', ascending=False)
        idcg = idcg_funct(poolActual['relevancia'])
        ndcg.append(dcg/idcg)

        #ERR
        er = err_funct(consulActual)
        err.append(er)

        #RBP
        val = (1-p) * rbp_funct(consulActual)
        rbp.append(val)

    #precision
    precisionFinal = np.mean(precision)
    print(f'Precision: {precisionFinal}')
    
    #recall
    recallFinal = np.mean(recall)
    print(f'Recall: {recallFinal}')

    #media armonica
    mediaArmonica = (2 * precisionFinal * recallFinal) / (precisionFinal + recallFinal)
    print(f'Media Armonica: {mediaArmonica}')

    #MAP
    print(f'MAP: {np.mean(map)}')

    #MRR
    print(f'MRR: {np.mean(mrr)}')

    #nDCG    
    print(f'nDCG: {np.mean(ndcg)}')

    #ERR
    print(f'ERR: {np.mean(err)}')

    #RBP
    print(f'RBP: {np.mean(rbp)}')


calcularMetricas("google")
calcularMetricas("duckduckgo")
calcularMetricas("ecosia")
calcularMetricas("bing")