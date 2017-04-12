# Toy benchmark program

'''

'''

from nnet.ranger import *
from nnet.trans_func import *
from nnet.stridelayers import *
from time import time

# PARAMETERS

nreps = 50
N = [(28, 28), (24, 24)]
S = (1,1)
M = [1, 20]
bs = 10
eta = .1

# INPUT

MN0 = np.hstack( (M[0], N[0]) )
MN1 = np.hstack( (M[1], N[1]) )

bsmn0 = np.hstack( (bs, MN0) )
bsmn1 = np.hstack( (bs, MN1) )

X = (np.arange(np.prod(bsmn0))/np.prod(bsmn0) - 0.5).reshape(bsmn0)
Y = (np.arange(np.prod(bsmn1))/np.prod(bsmn1) - 0.5).reshape(bsmn1) 

#window[i] = stride[i] * (1 - dims[i]) + input_dims[i]

window = np.array(S) * (1 - np.array(N[1])) + np.array(N[0]) 
Window = np.hstack( (M[1], window) )
W = (np.arange(np.prod(Window))/np.prod(Window) - 0.5).reshape(Window)
u = np.tile(0., (M[1], 1, 1))
Wo = np.copy(W)
uo = np.copy(u)


# PROCESSING

# Ranger

stride = strider(bsmn0, window, S)
Stride = stride;
if M[1] > M[0]:
  rep = np.hstack( [1, M[1], np.ones(len(N[0]), dtype = int)] )
  Stride = np.tile(Stride, rep)  + (np.arange(M[1], dtype = int) * (stride.max()+1)).reshape(rep)

i = 0
W = np.copy(Wo)
u = np.copy(uo)
B = None

t = time()
for h in range(nreps):
  x = np.take(X, stride)
  Z = (np.einsum('hijk,ik->hij', x, W.reshape((M[1], -1))) + u.reshape((M[1], 1))).reshape(bsmn1)

  A = sigval(Z)
  E = A - Y
  C = .5 * np.sum(E**2.)
  D = E * sigder(Z, A)

  d = D.reshape((bs, M[1], np.prod(N[1])))
  G = np.einsum('hijk,hik->hij', np.swapaxes(x, 2, 3), d).reshape(np.hstack((bs, Window)))

  """
  B = np.zeros(bs*M[1]*np.prod(N[0]), dtype = float)
  b = D.reshape((bs, M[1], np.prod(N[1]), 1)) * W.reshape( (1, M[1], 1, np.prod(window)) )
  np.add.at(B, Stride, b)
  B = B.reshape(np.hstack((bs, M[1], N[0])))
  """

  '''
  if i < s-1:
    W -= eta * G
    u -= eta * np.sum(np.sum(D, axis = 2), axis = 1)
  '''
print("Stride - Time taken: " + str(time()-t) + " s")

Z0, D0, G0, B0, C0 = np.copy(Z), np.copy(D), np.copy(G), np.copy(B), np.copy(C)


W = np.copy(Wo)
u = np.copy(uo)

Gi = np.array(N[1]) * np.array(S) - 1
Gj = Gi + window

t = time()

for h in range(nreps):
  w = reverse(W, [1, 2]).reshape(np.hstack( (1, Window) ))
  Z = subsample(conv(X, w, 'valid', axes = [2,3]), S) + u.reshape((1, M[1], 1, 1))
  A = sigval(Z)
  E = A - Y
  D = E * sigder(Z, A)
  d = reverse(intrapad(D, S), [2,3])
  G = subarray(conv(d, X, axes = [2,3]), Gi, Gj)
  #B = subarray(conv(d, w, axes = [2,3]), -1, -np.array(N[0])-1, -1)

  '''
  if i < s-1:
    W -= eta * G
    u -= eta * np.sum(np.sum(D, axis = 2), axis = 1)
  '''

Z1, D1, G1, B1, C1 = np.copy(Z), np.copy(D), np.copy(G), np.copy(B), np.copy(C)

print("Convolution -  Time taken: " + str(time()-t) + " s")

W = np.copy(Wo)
u = np.copy(uo)
t = time()
self = ConvLayer(M[1], N[1], S, "sigm")
self.setInput(M[0], N[0])
self.setMode(1)
self.setParams(W, u)
for h in range(nreps):
  self.forward(X)
  self.backward(self.output-Y)
print("Class.stride - Time taken: " + str(time()-t) + " s")

Z2, D2, G2, B2, C2  = np.copy(self.scores), np.copy(self.derivative), np.copy(self.gradient), np.copy(B), np.copy(C)

W = np.copy(Wo)
u = np.copy(uo)
t = time()
self.setMode(2)
self.setParams(W, u)
for h in range(nreps):
  self.forward(X)
  self.backward(self.output-Y)
print("Class.fft - Time taken: " + str(time()-t) + " s")

Z3, D3, G3, B3, C3  = np.copy(self.scores), np.copy(self.derivative), np.copy(self.gradient), np.copy(B), np.copy(C)
