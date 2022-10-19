import pandas as pd
import numpy as np

iterations = 300
r = 0.85
N = 5

pr = np.array([0.3,0.3,0.3,0.3,0.3])

M = np.array([[0,1,0,0,1],
              [0,0,0,0,0],
              [1,0,0,1,1],
              [0,0,1,0,0],
              [0,1,0,0,0]])

norm = np.sum(M, axis=1, keepdims=True)
M += (norm == 0)
norm = np.sum(M, axis=1, keepdims=True)
M = M / norm 

pr = pr/pr.sum()
e = np.array([0.5,0,0,0.5,0])

for i in range(iterations):
    term1 = np.matmul(pr.T, M) * r
    term2 = (1-r)*e
    pr = term1+term2

print(pr)