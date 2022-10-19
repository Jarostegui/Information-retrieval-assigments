import pandas as pd

qrels = pd.read_csv('datosOffline/qrels.txt', delimiter = "\t", names=["consulta", "inutil", "url", "relevancia"])

def juntarFicheros(buscador):
    results = pd.read_csv(f'datosOffline/{buscador}-results.txt', delimiter = "\t", names=["consulta", "inutil", "url", "posicion", "score", "buscador"])
    database = pd.merge(results, qrels, on=['consulta', 'inutil', 'url'], left_index=False)
    database.to_csv(f'datosOffline/database-{buscador}.txt', index=False)

juntarFicheros("google")
juntarFicheros("duckduckgo")
juntarFicheros("ecosia")
juntarFicheros("bing")