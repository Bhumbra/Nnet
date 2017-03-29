from nnet import supvlayers
from nnet import baselayers
import numpy as np
from time import time

# Multiple layer MNIST test

epochs = 3
bs = 10
eta = 0.1
Arch = [30, 30, 10] # 3 layer network 
transfunc = 'sigm'

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
net = [[]] * nArch

for h in range(nArch):
  if h < nArch - 1:
    net[h] = baselayers.BackLayer(Arch[h], transfunc)
  else:
    net[h] = supvlayers.CostLayer(Arch[h], transfunc)
  if h:
    net[h].setInput(net[h-1])
  else:
    net[h].setInput([input_data.shape[1]])
  self = net[h]

print("Training")
C = np.empty(epochs*int(np.ceil(len(input_data)/float(bs))), dtype = float)
k = 0
output = None

t = time()
for i in range(epochs):
  start, end = 0, bs
  done = end >= len(input_data)
  while not(done):
    for h in range(nArch):
      self = net[h]
      if not(h):
        output = self.forward(input_data[start:end])
      else:
        output = self.forward(output)
    for h in range(nArch-1, -1, -1):
      self = net[h]
      if h == nArch - 1:
        back  = self.backward(output_data[start:end])
      else:
        back  = self.backward(back)
      self.update(eta)
    C[k] = net[-1].cost_data
    start, end = end, end+bs
    done = end >= len(input_data)
    k += 1
  for h in range(nArch):
    self = net[h]
    if not(h):
      output = self.forward(test_input_data)
    else:
      output = self.forward(output)
  output_ind = np.argmax(output, axis = 1)
  accuracy = 100. * len(np.nonzero(output_ind == test_output_data)[0])/float(len(output))

  print("".join((str(time()-t),"s: ", str(i+1), '/', str(epochs), " accuracy: ", str(accuracy), "%" ) ) )
  
C = C[:k]

