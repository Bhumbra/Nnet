# A toy convolution test program to confirm broadcasting

from nnet.ranger import *
import numpy as np

A = np.tile(8., (1, 3, 4, 5))
B = np.tile(9., (2, 1, 4, 5))

C = conv(A, B, axes=[2, 3])

