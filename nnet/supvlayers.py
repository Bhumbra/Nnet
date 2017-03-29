"""
A module with classes for supervised learning.
"""

# Gary Bhumbra.

from nnet.baselayers import *
import nnet.cost_func as cost_func

class CostLayer (BackLayer):
  """
  This is a BackLayer with a cost function support for cost-based learning
  rules for supervised learning. It is therefore suitable for as an output
  layer. Note that CostLayer.backward(_output_data) directly compares
  activation outputs to _output_data.
  """
  cost_data = None
  costfunc = None # cost function string
  costfun = None  # cost function
  costder = None  # cost derivative

  def initialise(self, *args):
    BackLayer.initialise(self, *args)
    self.setCost()

  def setCost(self, *args):
    self.costfun = None
    self.costfun = None
    self.costder = None
    nargs = len(args)
    if not(nargs): 
      self.costfunc = 'quad'
      self.costfun, self.costder = cost_func.COST_FUNCTION_DERIVATIVE[self.costfunc]
    elif nargs == 1:
      if type(args[0]) is str:
        self.costfunc = args[0].lower()
        self.costfun, self.costder = cost_func.COST_FUNCTION_DERIVATIVE[self.costfunc]
      else:
        raise ValueError("self.setCost single inputs must be a string") 
    elif nargs == 2:
      self.costfunc, self.costfun, self.costder = 'cust', args[0], args[1]
    else:
      raise ValueError("Unexpected setCost(inputs arguments) specification.")

  def backward(self, _output_data = []):
    """
    _output_data refers to outputs to compare.
    outputs the back-propagated data
    can be combined with backLayer.update() to update parameters.
    """

    # Check batch-size 
    _output_data = np.asarray(_output_data)
    _batch_size = len(_output_data)
    if self.batch_size != _batch_size:
      self.setBatchSize(_batch_size)
      warnings.warn("CostLayer.backward() batch size not matched by Backlayer.forward().")

    # Test shape of output data and reshape according to expected input dimensionality
    if np.prod(_output_data.shape[1:]) != self.size:
      raise ValueError("Output data dimensions incommensurate with specified archecture")
    
    # Reshape data and calculate costs and derivatives
    self.output_data = _output_data.reshape([self.batch_size, self.size])
    self.cost_data = self.costfun(self.output_data, self.output)

    self.derivative = self.costder(self.output - self.output_data, self.scores, self.transder, self.output)
    #self.gradient = np.array([np.dot(self.derivative[i].reshape([self.size, 1]), self.input_data[i].reshape([1, self.input_size])) for i in range(self.batch_size)])
    self.gradient = np.einsum('ij,ik->ijk', self.derivative, self.input_data)

    # Now the gradient calculation and back-propagation
    self.back_data = np.dot(self.derivative, self.weight_coefs)

    return self.back_data

