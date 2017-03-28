"""
A contrived script to test code in layers.py
"""
import numpy as np
from nnet import baselayers 

X = np.random.uniform(size=(200, 5))

self = baselayers.baseLayer(5)
Y = self.forward(X)


