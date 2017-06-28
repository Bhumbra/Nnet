from nnet.layers import *
import numpy as np
from time import time

# Multiple layer MNIST test

epochs = 3
bs = 10
eta = 0.1
Arch = [30, 10] 
#Arch = [[-5,-5], (-2,-2), 10] 
#Arch = [[-5,-5], (-2,-2), 100, 10] 
maps = 20
dv = 2000
transfunc = 'sigm'
convmode = 1 # 0 default, 1 ranger, 2 fft
poolmode = 1 # 0 default, 1 ranger, 2 axes

print("Loading MNIST data...")
data = np.load('data/mnistpkl.npz', encoding = 'latin1')
training_data, validation_data, test_data = data['training_data'], data['validation_data'], data['test_data']

Data = training_data.swapaxes(0, 1)
input_data = np.hstack(Data[0]).T
output_data = np.hstack(Data[1]).T

TestData = test_data.swapaxes(0, 1)
test_input_data = np.hstack(TestData[0]).T
test_output_data = np.hstack(TestData[1]).T

print("...loaded.")
nArch = len(Arch)
Stack = [[]] * nArch

for h in range(nArch):
  arch = Arch[h]
  if type(arch) is list:
    Stack[h] = ConvLayer(maps, arch)
    Stack[h].setMode(convmode)
  elif type(arch) is tuple:
    Stack[h] = PoolLayer(maps, arch, transfunc)
    Stack[h].setMode(poolmode)
  elif h < nArch - 1:
    Stack[h] = FeedLayer(arch, transfunc)
  else:
    Stack[h] = CostLayer(arch, transfunc)
  if h:
    Stack[h].setInput(Stack[h-1])
  else:
    Stack[h].setInput((28, 28))

print("Training stack")
C = np.empty(epochs*int(np.ceil(len(input_data)/float(bs))), dtype = float)
k = 0
output = None

t = time()
for i in range(epochs):
  start, end = 0, bs
  done = end >= len(input_data)
  while not(done):
    if not(np.mod(start, dv)):
      print("".join(("   ", str(time()-t),"s: ", str(start), "/" + str(len(input_data)) ) ))
    #print("foo")
    for h in range(nArch):
      #print(h)
      self = Stack[h]
      if not(h):
        output = self.forward(input_data[start:end])
      else:
        output = self.forward(output)
    #print("bar")
    for h in range(nArch-1, -1, -1):
      #print(h)
      self = Stack[h]
      if h == nArch - 1:
        back  = self.backward(output_data[start:end])
      else:
        back  = self.backward(back)
      self.update(eta)
    #print("baz")
    C[k] = Stack[-1].cost_data
    start, end = end, end+bs
    done = end >= len(input_data)
    k += 1
  print("".join(("   ", str(time()-t),"s: ", str(start), "/" + str(len(input_data)) ) ))
  for h in range(nArch):
    self = Stack[h]
    if not(h):
      output = self.forward(test_input_data)
    else:
      output = self.forward(output)
  output_ind = np.argmax(output, axis = -1).ravel()
  accuracy = 100. * len(np.nonzero(output_ind == test_output_data)[0])/float(len(output))

  print("".join((str(time()-t),"s: ", str(i+1), '/', str(epochs), " accuracy: ", str(accuracy), "%" ) ) )
  
C = C[:k]

