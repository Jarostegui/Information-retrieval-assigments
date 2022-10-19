import pandas as pd
import numpy as np

def clicksPorConsulta(df):
    res = []
    for consul in df['qid'].unique():
        consulActual = df.loc[df['qid'] == consul]
        res.append(consulActual['count'].values)
    return np.mean(res)

def tasaDeAbandono(df):
    return df['qid'][df['count'] == 0].count() / len(df)

def maxRR(df):
    res = []
    for consul in df['qid'].unique():
        consulActual = df.loc[df['qid'] == consul]
        res.append(1/consulActual.iloc[0]['rank_clicked'])
    return np.mean(res)

def meanRR(df):
    res = []
    for consul in df['qid'].unique():
        consulActual = df.loc[df['qid'] == consul]
        for index, row in consulActual.iterrows():
            res.append(1/row['rank_clicked'])
    return np.mean(res)

def calcularMetricas(buscador):
    print("\n#####"+buscador+"#####")
    df = pd.read_csv(f'datosOnline/click-count-{buscador}.txt')
    df2 = pd.read_csv(f'datosOnline/click-log-{buscador}.txt')

    #Clicks por consulta
    cpc = clicksPorConsulta(df)
    print(f"Clicks por consulta: {cpc}")

    #Tasa de abandono
    ta = tasaDeAbandono(df)
    print(f'Tasa de abandono: {ta}')

    #Max RR
    maRR = maxRR(df2)
    print(f'Max RR: {maRR}')

    #Mean RR
    meRR = meanRR(df2)
    print(f'Mean RR: {meRR}')


calcularMetricas("ask")
calcularMetricas("bing")
calcularMetricas("duckduckgo")
calcularMetricas("google")