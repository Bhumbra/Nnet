"""
Class module for base layers that provide a foundation framework for both supervised
and unsupervised nodal networks.
"""

# Gary Bhumbra

import nnet.trans_func
import numpy as np
from dtypes import *

#-------------------------------------------------------------------------------
class baseLayer (object):
  """
  This is a convenience feedforward class for handling input, output, and archectural 
  dimensionality, with handling of activation functions and derivatives - but without 
  supervised or unsupersised learning functionality. It accommodates for multiple
  channels to allow the outer dimensionality demanded from feature maps. 
  
  It is camel-cased because it is not intended to be used natively but as a base class
  from an inheriting class. However it can be invoked:
  
  self = baseLayer(*args)
  
  where for args:

  #1 any integer specifies self.dims unless #2 is given - in which it specifies self.maps
  #2 any tuple, list, or ndarray specifies self.dims
  #3 any string (e.g. 'sigm', 'tanh', 'none', 'relu' specifies the transfer function)
  #4 a layer class specifies the relative input layer
 
  """
  dims = None     # layer dimensions within each feature map
  maps = None     # number of feature maps
  transfun = None # transfer function string
  transfer = None  # transfer function
  transder = None  # transfer derivative
  input_layer = None   # Input layer
  input_dims = None    # Input layer dimensnions
  input_maps = None    # Input layer maps
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
    self.dims = None     
    self.maps = None    
    self.step = None 

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
    for arg in args:
      if type(arg) is str:
        self.setTransfer(arg)
    for arg in args:
      if isinstance(arg, baseLayer):
        self.setInput(arg)

    # If no input dimensionality has been specified, default it
     
    if self.input_dims is None: self.setInput()

    # Initialise batch-size to 1

    self.setBatchSize()
    
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

    if self.dims is None: return

    # If no inputs has been specified, default dimensions and maps
    
    if self.input_dims is None: self.input_dims = self.dims
    if self.input_maps is None: self.input_maps = self.maps

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
    self.transfun = None
    self.transfer = None
    self.transder = None
    nargs = len(args)
    if ~args:
      pass
    elif nargs == 1:
      if type(args[0]) is str:
        self.transfun = args[0].lower()
        self.transfer, self.transder = trans_func.TRANSFER_FUNCTION_DERIVATIVE[self.transfun]
      else:
        raise ValueError("self.setTrasnfer single inputs must be a string") 
    elif nargs == 2:
      self.transfun, self.transfer, self.transder = 'cust', args[0], args[1]
    else:
      raise ValueError("Unexpected setTransfer(inputs arguments) specification.")

  def _setParamDims(self): 
    # a highly polymorphically mutable function guaranteed to be invoked at the right time

    if self.dims is None or self.input_dims is None: return
    if self.maps is None: self.maps = 1

    self.coef_shape = np.hstack( [self.maps, self.input_dims, self.dims] )
    self.offs_shape = np.hstack( [self.maps, 1, self.dims] )

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
      halfprod = 0.5 * float(self.maps * np.prod(self.dims)) 
      _weight_coefs = [-halfprod, halfprod]

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
    self.input_data = np.asarray(_input_data)
    _batch_size = len(_input_data)
    if self.batch_size != _batch_size:
      self.setBatchSize(_batch_size)

    # Reshape data with single-map dimensionality with a redundant singleton dimension
    if self.input_data.ndim == self.input_dims.ndim + 1: 
      if self.input_maps == 1:
        self.input_data = self.input_data.reshape(np.hstack((self.batch_size, self.input_maps, self.input_dims)))
   
    # Check input data dimensions are consistent with architecture
    if self.input_data.ndim != self.input_dims.ndim + 2:
      raise ValueError("Input data dimensionality incommensurate with specified archecture")
    else:
      if self.input_maps > 1 and self.maps > 1:
        if self.input_data.shape[1] != self.maps:
          raise ValueError("Input data maps incommensurate with specified archecture")
      elif np.any(self.input_data.shape[2:] != self.input_dims):
        raise ValueError("Input data dimensions incommensurate with specified archecture")
    

    # Now the arithmetic
    self.scores = np.inner(self.weight_coefs, self.input_data) + self.bias_offsets
    self.output = self.scores if self.transfer is None else self.transfer(self.scores)

    return self.output

#-------------------------------------------------------------------------------

