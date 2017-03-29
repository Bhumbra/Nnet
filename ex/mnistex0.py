from nnet import supvlayers
from nnet import baselayers
import numpy as np

# Single layer MNIST test

eta = 0.1
bs = 100
epochs = 10

print("Loading MNIST data")
data = np.load('data/mnistpkl.npz', encoding = 'latin1')
training_data, validation_data, test_data = data['training_data'], data['validation_data'], data['test_data']

Data = training_data.swapaxes(0, 1)
input_data = np.hstack(Data[0]).T
output_data = np.hstack(Data[1]).T

TestData = test_data.swapaxes(0, 1)
test_input_data = np.hstack(TestData[0]).T
test_output_data = np.hstack(TestData[1]).T

print("Constructing net")
self = supvlayers.CostLayer([output_data.shape[1]], 'sigm')
self.setInput([input_data.shape[1]])

print("Training net")
C = np.empty(epochs*int(np.ceil(len(input_data)/float(bs))), dtype = float)
k = 0
for i in range(epochs):
  start, end = 0, bs
  done = end >= len(input_data)
  while not(done):
    output = self.forward(input_data[start:end])
    back  = self.backward(output_data[start:end])
    C[k] = self.cost_data
    self.update(eta)
    start, end = end, end+bs
    done = end >= len(input_data)
    k += 1
  output = self.forward(test_input_data)
  output_ind = np.argmax(output, axis = 1)
  accuracy = 100. * len(np.nonzero(output_ind == test_output_data)[0])/float(len(output))

  print("".join( (str(i+1), '/', str(epochs), " accuracy: ", str(accuracy), "%" ) ) )
  
C = C[:k]

