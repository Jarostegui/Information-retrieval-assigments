from heapq import nlargest
import random
import pandas as pd
import numpy as np

class recommender():

    def __init__(self, k, N):
        self.k = k
        self.N = N
        self.usuarios = []
        self.items = []
        self.ratings = {}
        self.similitudes = {}
        self.porRecomendar = {}
        self.ratingsTest = {}
    
        
    def train(self, ficheroTrain, ficheroTest):
        dataset = open(ficheroTrain, 'r')
        lineas = dataset.read().splitlines()
        dataset.close()

        for linea in lineas:
            linea = linea.split("\t")
            usuario = int(linea[0])
            item = int(linea[1])
            rating = float(linea[2])
            if usuario not in self.usuarios:
                self.usuarios.append(usuario)
            if item not in self.items:
                self.items.append(item)
            if usuario in self.ratings:
                self.ratings[usuario][item] = rating
            else:
                diccionarioAux = {item: rating}
                self.ratings[usuario] = diccionarioAux
        for u in self.usuarios:
            self.porRecomendar[u] = list(set(self.items) - set(self.ratings[u].keys()))
            self.similitudes[u] = {}
            for v in self.usuarios:
                if u != v:
                    self.similitudes[u][v] = self.coseno(u, v)

        dataset = open(ficheroTest, 'r')
        lineas = dataset.read().splitlines()
        dataset.close()

        for linea in lineas:
            linea = linea.split("\t")
            usuario = int(linea[0])
            item = int(linea[1])
            rating = float(linea[2])
            if usuario in self.ratingsTest:
                self.ratingsTest[usuario][item] = rating
            else:
                diccionarioAux = {item: rating}
                self.ratingsTest[usuario] = diccionarioAux

        return

    def test(self):
        cont = 0
        MAE = 0
        RMSE = 0
        for u in self.ratingsTest.keys():
            for i in self.ratingsTest[u]:
                cont+=1
                pred = self.predecirRanking(u,i)
                real = self.ratingsTest[u][i]
                MAE += np.absolute(pred - real)
                RMSE += (pred - real) **2
                
        return MAE/cont, np.sqrt(RMSE/cont)

    def testRanking(self):
        precision = []
        recall = []
        for u in self.usuarios:
            aciertos = 0
            numRelevantes = 0
            ranking = self.chooseRanking(u)[:self.N]
            for i in ranking: 
                if i in self.ratingsTest[u] and self.ratingsTest[u][i] > 3.0:
                    aciertos += 1
            for i in self.ratingsTest[u]:
                if self.ratingsTest[u][i] > 3.0:
                    numRelevantes +=1
            precision.append(aciertos/self.N)
            recall.append(aciertos/numRelevantes)

        return np.mean(precision), np.mean(recall)

    def chooseRanking(self, usuario):
        result = {}
        usuariosSimilares = self.getTopK(usuario, self.k)
        for v in usuariosSimilares:
            for i in self.ratings[v].keys():
                if i in self.porRecomendar[usuario]:
                    result[i] = self.predecirRanking(usuario, i)
        return sorted(result, key=result.get, reverse=True)
    
    def predecirRanking(self, usuario, item):
        usuariosSimilares = self.getTopK(usuario, self.k)
        c = 0
        result = 0
        for v in usuariosSimilares:
            c += self.similitudes[usuario][v]
            if item in self.ratings[v].keys():
                result += self.similitudes[usuario][v] * self.ratings[v][item]
        return result/c

    def coseno(self, u, v):
        result = 0
        den1 = 0
        den2 = 0
        for i in self.ratings[u]:
            for j in self.ratings[v]:
                if i==j:
                    den1 += self.ratings[u][i]**2 
                    den2 += self.ratings[v][j]**2
                    result += self.ratings[u][i] * self.ratings[v][j]
        return  result / np.sqrt(den1*den2)

    def getTopK(self, usuario, k):
        keys = list(self.similitudes[usuario])
        kHighest = nlargest(k, keys, key=self.similitudes[usuario].get)
        return kHighest


R = recommender(k=2, N=5)
R.train('training-ratings.dat', 'test-ratings.dat')
MAE, RMSE = R.test()
Precision, Recall = R.testRanking()
print('K = '+str(R.k))
print('MAE: '+str(MAE))
print('RMSE: '+str(RMSE))
print('Precision@'+str(R.N)+': '+str(Precision))
print('Recall@'+str(R.N)+': '+str(Recall))

R = recommender(k=5, N=5)
R.train('training-ratings.dat', 'test-ratings.dat')
MAE, RMSE = R.test()
Precision, Recall = R.testRanking()
print('K = '+str(R.k))
print('MAE: '+str(MAE))
print('RMSE: '+str(RMSE))
print('Precision@'+str(R.N)+': '+str(Precision))
print('Recall@'+str(R.N)+': '+str(Recall))


R = recommender(k=10, N=5)
R.train('training-ratings.dat', 'test-ratings.dat')
MAE, RMSE = R.test()
Precision, Recall = R.testRanking()
print('K = '+str(R.k))
print('MAE: '+str(MAE))
print('RMSE: '+str(RMSE))
print('Precision@'+str(R.N)+': '+str(Precision))
print('Recall@'+str(R.N)+': '+str(Recall))