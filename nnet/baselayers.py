"""
Class module for base layers that provide a foundation framework for both supervised
and unsupervised nodal networks. 

OOP notation convention:

lowerCamelCase: base classes intended to inherited rather than instantiated.
UpperCamelCase: classes intended for instantiation.
self.lowerCamelCase: public class methods.
self._lowerCamelCase: protected class methods.
self.lowercase or self.snake_case: class properties.
_snake_case: nontrivial local variables that lose scope outside method.
"""

# Gary Bhumbra

import warnings
import numpy as np
import nnet.trans_func as trans_func
from dtypes import *

#-------------------------------------------------------------------------------
class baseLayer (object):
  """
  This is a foundation feedforward class for handling input, output, and archectural 
  dimensionality, supporting activation functions and derivatives, but without 
  intrinsic supervised or unsupersised learning functionality. It accommodates for 
  multiple channels to allow the outer dimensionality demanded from feature maps.
  
  It is lower camel-cased because it is not intended to be used natively but as a base 
  class for an inheriting class. However it can be invoked:
  
  self = baseLayer(*args)
  
  where for args:

  #1 any integer specifies self.dims unless #2 is given - in which it specifies self.maps
  #2 any tuple, list, or ndarray specifies self.dims
  #3 any string (e.g. 'sigm', 'tanh', 'none', 'relu' specifies the transfer function)
  #4 a layer class specifies the relative input layer

  Negative specifications in case #2 can be used for relative downsizing from preceding
  input layer.
 
  Note that feedfoward instances do not `own' the input_data. The input_data can be
  fed to the class using the baseLayer.feedforward(_input_data) method and the results 
  of the activation transfer function are returned by baseLayer.activate(). The two
  methods are combined in the baseLayer.forward(_input_data) method.

  """
  dims = None     # layer dimensions within each feature map
  maps = None     # number of feature maps
  size = None     # total number of nodes per feature map
  Size = None     # total number of nodes across all features maps
  transfunc = None # transfer function string
  transfer = None  # transfer function
  transder = None  # transfer derivative
  input_layer = None   # Input layer
  input_dims = None    # Input layer dimensnions
  input_maps = None    # Input layer maps
  input_size = None    # Input layer size
  input_Size = None    # Input layer size
  coef_shape = None     # Weighting coefficient dimensions
  offs_shape = None     # Bias offsets dimensions
  batch_size = None    # len(input_data)
  single_map = None    # flag to denote unitary map in present layer
  input_data = None    # X
  weight_coefs = None  # W
  bias_offsets = None  # u
  scores = None        # Z = WX + U (pre-non-linearity values)
  output = None        # A = transfer(Z) (post-non-linearity values)

  def __init__(self, *args):
    self.initialise(*args)

  def initialise(self, *args):
    self.setDims(*args)

    # If specified set input specification, otherwise default it
    for arg in args:
      if isinstance(arg, baseLayer):
        self.setInput(arg)

    if self.input_dims is None: self.setInput()

    # If specified set transfer function
    for arg in args:
      if type(arg) is str:
        self.setTransfer(arg)

    # Initialise batch-size to 0
    self.setBatchSize()

  def setDims(self, *args):

    self.dims = None     
    self.maps = None    

    # Initialise layer dimensions
    for arg in args:
      if isarray(arg):
        if self.dims is None:
          self.dims = np.atleast_1d(arg)
        else:
          raise ValueError("Unexpected number of array-like input arguments.")

    for arg in args:
      if isint(arg):
        if self.dims is None:
          self.dims = np.atleast_1d([arg])
        elif self.maps is None:
          self.maps = int(arg)
        else:
          raise ValueError("Unexpected number of integer input arguments.")

    # If dims has been initialised, default feature maps 
    if self.dims is not None:
      if self.maps is None:
        self.maps = 1

  def setInput(self, *args):
    """
    This sets the input dimensionality or input connecting class - not the data itself.

    Two ways:
      self.setInput(inputLayerInstance)
      self.setInput(inputLayerInstance.dims, inputLayerInstance.maps)

    An input_layer with no nodes but merely receptors is not an input layer
    """
    
    self.input_layer = None   
    self.input_dims = None    
    self.input_maps = None    
    self.input_size = None
    self.input_Size = None
    for arg in args:
      if isinstance(arg, baseLayer):
        self.input_layer = arg
        self.input_dims = arg.dims
        self.input_maps = arg.maps
      elif isarray(arg):
        if self.input_dims is None:
          self.input_dims = np.atleast_1d(arg)
        else:
          raise ValueError("Unexpected number of array-like input arguments.")
      elif isint(arg):
        if self.input_maps is None:
          self.input_maps = int(arg)
        else:
          raise ValueError("Unexpected number of integer input arguments.")
      else:
        raise ValueError("Unexpected input argument type.")

    # Swap input_dims and input_maps to correspond to convention of current dims and maps
    if self.input_dims is None:
      if self.input_maps is not None:
        self.input_dims = np.atleast_1d(self.input_maps)
        self.input_maps = 1
    elif self.input_maps is None:
      self.input_maps = 1

    # Return if insufficient information provided
    if self.dims is None: # If present layer has no dimensions, return
      return
    elif self.input_dims is None: # If input dims not given and relatively sized, return
      if np.min(self.dims) < 1: return

    # If no inputs has been specified, default dimensions and maps
    if self.input_dims is None: self.input_dims = self.dims
    if self.input_maps is None: self.input_maps = self.maps

    self.input_size = np.prod(self.input_dims)
    self.input_Size = self.input_maps * self.input_size 

    # Occam's razer should apply if input and current layer have single feature maps
    self.single_map = self.maps == 1

    # Initialise parameters
    self._setParamDims()
    self.setParams()

  def setTransfer(self, *args):
    """
    This sets the transfer function (default None)

    Two ways:
      self.setTransfer('sigm')
      self.setTransfer(transferFunction, transferDerivativeFunction

    - note that unless the derivative function is None, a second input of the
      derivative specifying pre-derivative values must be allowed even if
      never used.
    """
    self.transfunc = None
    self.transfer = None
    self.transder = None
    nargs = len(args)
    if not(nargs):
      pass
    elif nargs == 1:
      if type(args[0]) is str:
        self.transfunc = args[0].lower()
        self.transfer, self.transder = trans_func.TRANSFER_FUNCTION_DERIVATIVE[self.transfunc]
      else:
        raise ValueError("self.setTransfer single inputs must be a string") 
    elif nargs == 2:
      self.transfunc, self.transfer, self.transder = 'cust', args[0], args[1]
    else:
      raise ValueError("Unexpected setTransfer(inputs arguments) specification.")

  def _setParamDims(self): 
    # a highly polymorphically mutable function guaranteed to be invoked at the right time

    if self.dims is None or self.input_dims is None: return
    if self.maps is None: self.maps = 1

    neg_dims = self.dims < 1

    if len(self.dims) == len(self.input_dims):
      if np.any(self.input_dims < 1):
        warnings.warn("Relative dimension specification ambiguous in relation to input specification.")
      else:
        self.dims[neg_dims] += self.input_dims[neg_dims]
    elif np.any(neg_dims):
      raise ValueError("Relative dimension specfication incommensurate with previous layer")

    self.size = np.prod(self.dims)
    self.Size = self.maps * self.size
    
    if self.single_map:
      self.coef_shape = np.atleast_1d([self.size, self.input_Size])
      self.offs_shape = np.atleast_1d([1, self.size])
    else:
      self.coef_shape = np.atleast_1d([self.maps, self.size, self.input_size])
      self.offs_shape = np.atleast_1d([self.maps, 1, self.size])

  def setParams(self, _weight_coefs = 1., _bias_offsets = 0.):
    self.weight_coefs = None
    self.bias_offsets = None
   
    if type(_weight_coefs) is np.ndarray:
      self.weight_coefs = _weight_coefs
      if np.any(self.coef_shape != self.weight_coefs.shape):
        raise ValueError("Coefficient array of unexpected dimensions.")

    if type(_bias_offsets) is np.ndarray:
      self.bias_offsets = _bias_offsets
      if np.any(self.offs_shape != self.bias_offsets.shape):
        raise ValueError("Offset array of unexpected dimensions.")

    if self.weight_coefs is None:
      self.weight_coefs = _weight_coefs * (np.random.uniform(size = self.coef_shape) - 0.5)

    if self.bias_offsets is None:
      self.bias_offsets = _bias_offsets * (np.random.uniform(size = self.offs_shape) - 0.5)

  def setBatchSize(self, _batch_size = 0):
    """ 
    This not only specifies the batch-size in a data-absent manner but allows overloaded polymorphism in inheriting
    classes to adjust batch-size-sensitive arrays and methods.
    """
    self.batch_size = int(_batch_size) if isint(_batch_size) else len(_batch_size)

  def forward(self, _input_data = []):
    """
    Convenience function combining two lines of code:

    self.feedforward(_input_data)
    return self.activate()
    """

    self.feedforward(_input_data)
    return self.activate()

  def feedforward(self, _input_data = []):
    """
    Forward-propagates data through layer to evaluate scores: Z = (W * X.T).T + u
    """

    _input_data = np.asarray(_input_data)

    # Update self.batch_size if necessary
    if self.batch_size != len(_input_data):
      self.setBatchSize(_input_data)

    # Test shape of input data and reshape according to expected input dimensionality
    if np.prod(_input_data.shape[1:]) != self.input_Size:
      raise ValueError("Input data dimensions incommensurate with specified archecture")
    
    # Forward operaion

    if self.single_map: # mysteriously, dot is faster than inner
      self.input_data = _input_data.reshape([self.batch_size, self.input_Size])
      #self.scores = np.inner(self.weight_coefs, self.input_data).T + self.bias_offsets
      self.scores = np.dot(self.weight_coefs, self.input_data.T).T + self.bias_offsets
    else:
      self.input_data = _input_data.reshape([self.batch_size, self.input_maps, self.input_size])
      self.scores = np.einsum("kij,hij->hki", self.weight_coefs, self.input_data) + self.bias_offsets

    return self.scores

  def activate(self, _scores = None):
    """
    Evaluates transfer function from scores: A = \sigma(Z)
    """
    if _scores is not None:
      _output = _scores if self.transfer is None else self.transfer(_scores)
      return _output
    self.output = self.scores if self.transfer is None else self.transfer(self.scores)
    return self.output
    
#-------------------------------------------------------------------------------
class FeedLayer (baseLayer):
  """
  This is a generic feedforward + feedback class inheriting from feedlayer but
  with the additional feedback and backpropagation functionality of backward(). 
  It is suitable for use as either a first or hidden fully-connected layer, 
  but not output layer as it supports neither unsupervised or supervised learning
  rules. 
  
  It is instantiated in the same way as baseLayer: see help(baseLayer). 

  This class also offers FeedLayer.update(_eta = 1) which updates the coefficient 
  weights and bias offsets according to stochastic gradient descent scaled by the 
  batch size.

  Note that FeedLayer instances do not `own' the output_data. The output_data can be
  fed to the class using the FeedLayer.feedback(_output_data) method and the results 
  of the backpropagation are returned by FeedLayer.backpropagate(). The two
  methods are combined in the FeedLayer.backward(_input_data) method.

  """
  output_data = None
  derivative = None # = d(cost)/d(scores)       = [d(cost)/d(output)]   * [d(output)/d(scores)]
  gradient = None   # = d(cost)/d(weight_coefs) = [d(cost)/d(scores)].T * input_data
  back_data = None  # = d(cost)/d(input_data)   = [d(cost)/d(scores)]   * weight_coefs.T
  eta = None        # Learning rate i.e. constant prop. for weight and bias updates
  coef_delta = None # change to weight_coefs
  offs_delta = None # change to bias_offsets

  def initialise(self, *args):
    baseLayer.initialise(self, *args)
    self.update(1.) # default learning rate

  def backward(self, _output_data = []):
    """
    Convenience function combining two lines of code:

    self.feedback(_output_data)
    return self.backpropagate()
    """
    self.feedback(_output_data)
    return self.backpropagate()

  def feedback(self, _output_data = []):
    """
    Feeds `errors' through layer to compute the derivate: D = E o \sigma^\prime(Z)
    - also calculates gradients for weight-coefficient updates: G = (D * X).T
    
    Note that output_data refers to `errors' _not_ and does not compute costs.   
    
    """
    _output_data = np.asarray(_output_data)

    # Check batch-size 
    _batch_size = len(_output_data)
    if self.batch_size != len(_output_data):
      self.setBatchSize(_output_data)
      warnings.warn("FeedLayer.backward() batch size not matched by Backlayer.forward().")

    # Test shape of output data and reshape according to expected input dimensionality
    if np.prod(_output_data.shape[1:]) != self.Size:
      raise ValueError("Output data dimensions incommensurate with specified archecture")
    
    # Reshape data and calculate derivatives

    if self.single_map:
      self.output_data = _output_data.reshape([self.batch_size, self.size])
      self.derivative = self.output_data if self.transder is None else self.output_data * self.transder(self.scores, self.output)
      #self.gradient = np.einsum('ik,il->ikl', self.derivative, self.input_data)
      self.gradient = self.derivative.reshape((self.batch_size,self.size, 1)) * self.input_data.reshape(self.batch_size,1, self.input_size)
    else:
      self.output_data = _output_data.reshape([self.batch_size, self.maps, self.size])
      self.derivative = self.output_data if self.transder is None else self.output_data * self.transder(self.scores, self.output)
      self.gradient = np.einsum('ijk,ijl->ijkl', self.derivative, self.input_data)

    return self.derivative

  def backpropagate(self):
    """
    Backpropagates derivatives to calculate back-propagated `errors': B = D * W
    
    Note that no backpropagation calculation is performed without a known input layer.
    
    """

    # Now the gradient calculation and back-propagation
    if self.input_layer is None: return self.back_data
    self.back_data = np.dot(self.derivative, self.weight_coefs)
    if self.single_map and self.input_maps == 1: return self.back_data # simplicity propagated
    self.back_data = self.back_data.reshape(np.hstack([self.batch_size, self.input_maps, self.input_dims]))
    return self.back_data
  
  def update(self, _eta = None):
    pass
    """
    Input eta specifies update constant of proportionality (i.e. `learning' rate).
    Returns Delta_weights, Delta_biases.
    """

    if _eta is not None: self.eta = _eta
    if self.derivative is None: return None, None

    # Check batch-size 
    if self.batch_size != len(self.derivative):
      self.setBatchSize(self.derivative)
      warnings.warn("FeedLayer.update() batch size not matched by FeedLayer.backward().")

    self.coef_delta = np.zeros(self.coef_shape, dtype = float)
    self.offs_delta = np.zeros(self.offs_shape, dtype = float)

    if not(self.batch_size): 
      return self.coef_delta, self.offs_delta

    # Derivative first
   
    if np.prod(self.offs_shape) == self.maps:
      self.offs_delta = -self.eta * np.sum(np.mean(self.derivative, axis = 0).reshape((self.maps, -1)), axis = 1).reshape(self.offs_shape)
    else:
      self.offs_delta = -self.eta * np.mean(self.derivative, axis = 0).reshape(self.offs_shape)

    # Now the gradient
    if self.gradient is not None:
      self.coef_delta = -self.eta * np.mean(self.gradient, axis = 0).reshape(self.coef_shape)

    self.setParams(self.weight_coefs + self.coef_delta, self.bias_offsets + self.offs_delta)

    return self.coef_delta, self.offs_delta

#-------------------------------------------------------------------------------

