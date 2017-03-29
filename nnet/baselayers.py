"""
Class module for base layers that provide a foundation framework for both supervised
and unsupervised nodal networks.
"""

# Gary Bhumbra

import warnings
import numpy as np
import nnet.trans_func
from dtypes import *

#-------------------------------------------------------------------------------
class feedLayer (object):
  """
  This is a convenience feedforward class for handling input, output, and archectural 
  dimensionality, with handling of activation functions and derivatives - but without 
  supervised or unsupersised learning functionality. It accommodates for multiple
  channels to allow the outer dimensionality demanded from feature maps. 
  
  It is camel-cased because it is not intended to be used natively but as a base class
  for an inheriting class. However it can be invoked:
  
  self = baseLayer(*args)
  
  where for args:

  #1 any integer specifies self.dims unless #2 is given - in which it specifies self.maps
  #2 any tuple, list, or ndarray specifies self.dims
  #3 any string (e.g. 'sigm', 'tanh', 'none', 'relu' specifies the transfer function)
  #4 a layer class specifies the relative input layer
 
  Note that feedfoward instances do not `own' the input_data.
  """
  dims = None     # layer dimensions within each feature map
  maps = None     # number of feature maps
  size = None     # total number of nodes
  transfunc = None # transfer function string
  transfer = None  # transfer function
  transder = None  # transfer derivative
  input_layer = None   # Input layer
  input_dims = None    # Input layer dimensnions
  input_maps = None    # Input layer maps
  input_size = None    # Input layer size
  coef_shape = None     # Weighting coefficient dimensions
  offs_shape = None     # Bias offsets dimensions
  batch_size = None    # len(input_data)
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
      if isinstance(arg, feedLayer):
        self.setInput(arg)

    if self.input_dims is None: self.setInput()

    # If specified set transfer function
    for arg in args:
      if type(arg) is str:
        self.setTransfer(arg)

    # Initialise batch-size to 1
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

    self.size = self.maps * np.prod(self.dims)
    
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
    for arg in args:
      if isinstance(arg, BaseLayer):
        self.input_layer = arg
        self.input_dims = arg.dims
        self.input_maps = arg.maps
      elif isarray(arg):
        if self.input_dims is None:
          self.input_dims = np.atleast_1d(arg)
        else:
          raise ValueError("Unexpected number of array-like input arguments.")
      elif isint(arg):
        self.input_maps = int(arg)

    # If present layers has no dimensions, return
    if self.dims is None: 
      self.input_size = 1 if self.input_maps is None else self.input_maps
      if self.input_dims is not None: self.input_size *= np.prod(self.input_dims)
      return

    # If no inputs has been specified, default dimensions and maps
    
    if self.input_dims is None: self.input_dims = self.dims
    if self.input_maps is None: self.input_maps = self.maps

    self.input_size = self.input_maps * np.prod(self.input_dims)

    # Initialise parameters

    self._setParamDims()
    self.initParams()

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
    if ~args:
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

    self.coef_shape = np.atleast_1d([self.size, self.input_size])
    self.offs_shape = np.atleast_1d(self.size)

    if self.maps > 1:
      warnings.warn("baseLayer.forward() does not support multiple features maps.")

  def initParams(self, _weight_coefs = None, _bias_offsets = 0.):
    self.weight_coefs = None
    self.bias_offsets = None
   
    if type(_weight_coefs) is np.ndarray:
      self.weight_coefs = _weight_coefs
      if np.any(self.coef_shape != self.weight_coefs):
        raise ValueError("Coefficient array of unexpected dimensions.")

    if type(_bias_offsets) is np.ndarray:
      self.bias_offsets = _bias_offsets
      if np.any(self.offs_shape != self.bias_offsets):
        raise ValueError("Offset array of unexpected dimensions.")

    if _weight_coefs is None:
      invprod = .5 / float(np.maximum(1, self.maps * np.prod(self.dims)) )
      _weight_coefs = [-invprod, invprod]

    if self.weight_coefs is None:
      _weight_coefs = np.atleast_1d(_weight_coefs)
      if len(_weight_coefs) == 1:
        self.weight_coefs = np.tile(_weight_coefs[0], self.coef_shape)
      elif len(_weight_coefs) == 2:
        self.weight_coefs = np.diff(_weight_coefs) * (np.random.uniform(size = self.coef_shape)-_weight_coefs[0])
      else:
        raise Value("Unknown weight coefficient initialisation specification.")

    if self.bias_offsets is None:
      _bias_offsets = np.atleast_1d(_bias_offsets)
      if len(_bias_offsets) == 1:
        self.bias_offsets = np.tile(_bias_offsets[0], self.offs_shape)
      elif len(_bias_offsets) == 2:
        self.bias_offsets = np.diff(_bias_offsets) * (np.random.uniform(size = self.offs_shape)-_bias_offsets[0])
      else:
        raise Value("Unknown bias offset initialisation specification.")

  def setBatchSize(self, _batch_size = 1):
    """ 
    This not only specifies the batch-size in a data-absent manner but allows overloaded polymorphism in inheriting
    classes to adjust batch-size-sensitive arrays allocated in memoroy
    """
    self.batch_size = int(_batch_size) if isint(_batch_size) else len(_batch_size)

  def forward(self, _input_data = []):

    # Update self.batch_size if necessary
    _input_data = np.asarray(_input_data)
    _batch_size = len(_input_data)
    if self.batch_size != _batch_size:
      self.setBatchSize(_batch_size)

    # Test shape of input data and reshape according to expected input dimensionality
    if np.prod(_input_data.shape[1:]) != self.input_size:
      raise ValueError("Input data dimensions incommensurate with specified archecture")
    
    self.input_data = _input_data.reshape([self.batch_size, self.input_size])
      
    # Now the matrix multiplication, offsetting, and transfer function
    self.scores = np.dot(self.weight_coefs, self.input_data.T).T + self.bias_offsets
    self.output = self.scores if self.transfer is None else self.transfer(self.scores)

    return self.output

#-------------------------------------------------------------------------------
class BackLayer (feedLayer):
  """
  This is a convenience feedforward + feedback class inheriting from feedlayer but
  with the additional facility of backward(). It is suitable for use as either a
  first or hidden fully-connected layer, but not output layer.
  
  It is instantiated in the same way as feedLayer. 

  This class also offers BackLayer.update(eta = 1) which updates the coefficient weights
  and bias offsets according to stochastic gradient descent scaled by the batch size.

  Note that BackLayer instances do not `own' the output_data.
  """
  output_data = None
  derivative = None # = d(cost)/d(scores)       = [d(cost)/d(output)]   * [d(output)/d(scores)]
  gradient = None   # = d(cost)/d(weight_coefs) = [d(cost)/d(scores)].T * input_data
  back_data = None  # = d(cost)/d(input_data)   = [d(cost)/d(scores)]   * weight_coefs.T
  eta = None        # Learning rate i.e. constant prop. for weight and bias updates
  coef_delta = None # change to weight_coefs
  offs_delta = None # change to bias_offsets

  def initialise(self, *args):
    feedLayer.initialise(self, *args)
    self.update(1.) # default learning rate

  def backward(self, _output_data = []):
    """
    Propagates errors through layer. Note:

    _output_data refers to `errors' _not_ known outputs.
    outputs the back-propagated data
    can be combined with backLayer.update() to update parameters.
    """

    # Check batch-size 
    _output_data = np.asarray(_output_data)
    _batch_size = len(_output_data)
    forwardMatches = self.batch_size == _batch_size
    if ~forwardMatches:
      self.setBatchSize(_batch_size)
      warnings.warn("BackLayer.backward() batch size not matched by Backlayer.forward().")

    # Test shape of output data and reshape according to expected input dimensionality
    if np.prod(_output_data.shape[1:]) != self.size:
      raise ValueError("Output data dimensions incommensurate with specified archecture")
    
    # Reshape data and calculate derivatives
    self.output_data = _output_data.reshape([self.batch_size, self.size])
    if forwardMatches:
      self.derivative = self.output_data if self.transfer is None else self.transder(self.output_data, self.scores)
      self.gradient = self.derivative.T * self.input_data
    else: 
      self.derivative = self.output_data if self.transfer is None else self.transder(self.output_data)
      self.gradient = None # cannot be calculated due mismatched input/output batch sizes

    # Now the gradient calculation and back-propagation
    self.back_data = self.derivative * self.coef_weights.T

    return self.back_data
  
  def update(self, _eta = None):
    """
    Input eta specifies update constant of proportionality (i.e. `learning' rate).
    Returns Delta_weights, Delta_biases.
    """

    if _eta is not None: self.eta = _eta
    if self.coef_shape is None: return None, None
    if self.offs_shape is None: return None, None
    if self.derivate is None: return None, None

    # Check batch-size 
    _batch_size = len(self.derivative)
    if self.batch_size != self.derivative:
      self.setBatchSize(_batch_size)
      warnings.warn("BackLayer.update() batch size not matched by BackLayer.backward().")

    self.coef_delta = np.zeros(self.coef_shape, dtype = float)
    self.offs_delta = np.zeros(self.offs_shape, dtype = float)

    if ~self.batch_size: 
      return self.coef_delta, self.offs_delta

    # Derivative first
    if self.derivative is not None:
      self.offs_delta = -self.eta * np.mean(self.derivative, axis = 0).reshape(self.offs_shape)

    # Now the gradient
    if self.gradient is not None:
      self.coef_delta = -self.eta * np.mean(self.gradient, axis = 0).reshape(self.coef_shape)

    self.weight_coefs += self.coef_delta
    self.bias_offsets += self.offs_delta

    return self.coef_delta, self.offs_delta

#-------------------------------------------------------------------------------

