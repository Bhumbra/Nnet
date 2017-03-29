"""
A contrived script to test code in baselayers.py
"""
import numpy as np
from nnet import baselayers 

X = np.random.uniform(size=(200, 5))

self = baselayers.feedLayer(5)
Y = self.forward(X)

print(Y[0])
print(np.dot(np.matrix(self.weight_coefs), np.matrix(X).T).T[0])

