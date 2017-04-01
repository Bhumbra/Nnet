"""
A module for layer classes for strided operations with feature maps. These can be used for supervised
and unsupervsed learning. For classical convolutional networks, the classes ConvLayer and PoolLayer
will be most useful here.

Here I use `kernel' to refer to the weight coefficient or pooling size, and `filter' to be the actual 
convolution filter (i.e. rotated weight_coef).
"""

# Gary Bhumbra.

from nnet.baselayers import *

CONVMODE_DEFAULT = 0
CONVMODE_RANGER = 1
CONVMODE_FFT = 2

POOLMODE_DEFAULT = 0
POOLMODE_RANGER = 1
POOLMODE_AXES = 2

#-------------------------------------------------------------------------------
class strideLayer(BackLayer):
  """
  This is a BackLayer with explicit support for feature maps each containing arbitrary dimensions. No
  special multidimensional operations are supported in forward() or backward() such this is a base
  class intended to be inheriting to other classes, notably ConvLayer and PoolLayer. Since it is not 
  intended to be directly instanstiated itself, it is camel-cased.

  Initialisation of this and inheriting classes with specification dimensions differs with that of 
  previous classes in two ways:
  
  1. Negative indexing of node dimensions specifies kernel dimensions.
  2. A second array-like input is interpreted as a stride specification. Zero elements of this stride 
     specification match the stride specification to kernel size in the corresponding axis. Negative 
     (or zero) elements here can be used to specify the stride relative to the kernel only if the
     kernel size was specified by negative indexing in the first array.
     
  Note strideLayer.coef_shape concatenates maps with kernel size and strideLayer.single_map is ignored.

  """

  def_stride = [1, 1] # default [neg_dim stride, pos_dim stride] - which can be adjusted by other classes (0 matches kernel)
  single_map = None   # ignored by this function
  stride = None
  kernel = None  # size of kernel (i.e. coef_shape) - this term is less confusing for pooling
  filter = None

  def initialise(self, *args):
    BackLayer.initialise(self, *args)

  def setDims(*args):
    BackLayer.setDims(*args)
    self.stride = None

    array_count = 0
    for arg in args:
      if isarray(arg):
        array_count += 1
        if array_count == 2:
          self.stride = np.atleast_1d(arg)

  def _setParamDims(self): 
    # this overloads feedLayer._setParamDims and is guaranteed to be invoked at the right time

    if self.dims is None or self.input_dims is None: return
    if self.maps is None: self.maps = 1
    if self.strides is None: 
      self.strides = np.tile(self.def_stride[0], len(self.dims))
      self.strides[self.dims > 0] = self.def_stride[1]

    if len(self.dims) != len(self.strides):
      raise ValueError("Dimensionality in node dimension and stride specification incommensurate")

    """
    4 combinations - and do in this order of increasing complexity!
    positive dims, negative kernel, negative strides -> disallowed
    positive dims, negative kernel, positive strides -> evaluate kernel
    negative dims, positive kernel, negative strides -> evaluate strides then dims
    negative dims, positive kernel, positive strides -> evaluate dims then swap

    Dimension equation: 
      kernel[i] = stride[i] * (1 - dims[i]) + input_size[i]
                or
      dims[i]   = 1 + (input_size[i] - kernel[i])/stride[i]
                or
      stride[i] = (kernel[i] - input_size[i]) / (1 - dims[i])
    """
    
    self.kernel = np.maximum(np.zeros(len(self.dims), dtype = int), -self.dims) 
    
    pos_kernel, pos_strides = self.kernel > 0,            self.strides > 0
    neg_kernel, neg_strides = np.logical_not(pos_kernel), np.logical_not(pos_strides)

    # positive dims, negative kernel, negative strides -> disallowed
    cases = np.logical_and(neg_kernel, neg_strides)
    if np.any(cases):
      raise ValueError("Negative indexing for strides requires negative indexing of dimension specification")

    # positive dims, negative kernel, positive strides -> evaluate kernel
    cases = np.logical_and(neg_kernel, pos_strides)
    self.kernel[cases] = self.stride[cases] * (1 - self.dim[cases]) + self.input_size[cases]
    
    # negative dims, positive kernel, negative strides -> evaluate strides then dims
    cases = np.logical_and(pos_kernel, neg_strides)
    self.strides[cases] += self.kernel[cases]
    self.dims[cases] = 1 + (self.input_size[cases] - self.kernel[cases]) / self.stride[cases]
    
    # negative dims, positive kernel, positive strides -> evaluate dims then swap
    cases = np.logical_and(pos_kernel, pos_strides)
    self.dims[cases] = 1 + (self.input_size[cases] - self.kernel[cases]) / self.stride[cases]
    self.dims[cases], self.kernel[cases] = self.kernel[cases], self.strides[cases]

    self.coef_shape = np.hstack((self.maps, self.kernel))
    self.offs_shape = np.atleast_1d(self.maps, np.ones(self.size, dtype = int)) # one offset per map

  def setBatchSize(self, _batch_size = 1):
    """ 
    This class overloads this function to block batch usage because this class shouldn't have any.
    """
    #self.batch_size = int(_batch_size) if isint(_batch_size) else len(_batch_size)

    raise TypeError("Batch operations not available directly from strideLayer instances")

#-------------------------------------------------------------------------------

class PoolLayer(strideLayer):
  """
  This is a strideLayer with explicit support for pooling, with Initialisation specification
  identical to strideLayers except default stride specification is set to the window size for negative
  indexed dimension specification. PoolLayer.setMode(pool_mode) sets:

  pool_mode = POOLMODE_DEFAULT : uses default
  pool_mode = POOLMODE_RANGER  : uses fancy indexing
  pool_mode = POOLMODE_AXES    : uses re-axes pooling unless overlapping


  """
  def_stride = [0, 1]
  pool_mode = None     # user-specified mode
  mode = None          # mode actually used
  stride = None        # flattened indices
  input_pool = None    # input for pooling
  ran_pooled = None    # index for pooled range
  arg_pooled = None    # argument output from pooling

  def __init__(self, *args):
    self.initialise(*args)
    self.forward_call = {POOLMODE_RANGER:self.forward_ranger, POOLMODE_AXES:self.forward_axes}

    # Back-propagation still uses ranger.strider so we do not polymorph self.backward

  def initialise(self, *args):
    strideLayer.initialise(self, *args)
    self.setMode()

  def setMode(self, _pool_mode = None):
    if _pool_mode is not None: 
      self.pool_mode = _pool_mode
    elif self.pool_mode is None:
      self.pool_mode = POOLMODE_DEFAULT

  def setBatchSize(self, _batch_size = 0): # this formally assigns the mode and index
    """ 
    This class overloads this function and establishes actual pooling mode.
    """
    self.batch_size = int(_batch_size) if isint(_batch_size) else len(_batch_size)
    if not(self.batch_size): return

    # Pooling mode defaults to ranger
    
    self.mode = POOLMODE_RANGER
    if np.all(self.strides == self.kernel):
      if self.pool_mode != POOLMODE_RANGER:
        self.mode = POOLMODE_AXES

    # Whatever mode, we need self.stride for the back-propagation
    self.stride = strider(np.hstack((self.batch_size, self.input_maps, self.dims)),
                          np.hstack((self.maps, self.kernel)), self.strides)

    if self.mode == POOLMODE_RANGER:
      self.ran_pooled = np.range(len(self.stride), dtype = int)
    else:
      stridesize = len(self.stride)/(self.batch_size, self.maps)
      inner_ind = np.tile(np.arange(stridesize).reshape(1, 1, stridesize), (self.batch_size, self.maps, 1))
      middl_ind = np.tile(np.arange(self.maps).reshape(1, self.maps, 1), (self.batch_size, 1, stridesize))
      outer_ind = np.tile(np.arange(self.batch_size).reshape(self.batch_size, 1, 1), (1, self.maps, stridesize))
      self.ran_pool = np.hstack( (outer_ind, middl_ind, inner_ind) )

  def forward(self, _input_data = []):
    if self.batch_size != len(_input_data):
      self.setBatchSize(_input_data)
    if self.batch_size:
      return self.forward_call[self.mode](_input_data)

  def forward_ranger(self, _input_data = []):
    self.input_data = _input_data
    self.input_pool = np.take(self.input_data, self.stride)
    self.arg_pooled = np.argmax(self.input_pool, axis = -1)
    self.scores = np.reshape(self.input_pool[self.ran_pooled][self.arg_pooled],
                             np.hstack((self.batch_size, self.maps, self.dims)))

    return self.activate() 

  def forward_axes(self, _input_data = []):
    self.input_data = _input_data
    self.input_pool = poolaxes(self.input_data. self.kernel)
    self.arg_pooled = np.argmax(self.input_pool, axis = -1)
    self.scores = self.input_pool[tuple(np.hstack(self.ran_pooled, self.arg_pooled))]    
    return self.activate() 

#-------------------------------------------------------------------------------
class ConvLayer(strideLayer):
  """
  This is a strideLayer with explicit support for convolution, with initialisation specification
  identical to strideLayers except default stride specification is set 1. 
  ConvLayer.setMode(conv_mode) sets:

  conv_mode = CONVMODE_DEFAULT : uses default
  conv_mode = CONVMODE_RANGER  : uses fancy indexing
  conv_mode = CONVMODE_FFT     : uses Fourier-accelerated convolution.

  """
  def_stride = [1, 1]
  conv_mode = None     # user-specified mode
  mode = None          # mode actually used
  input_conv = None    # input for convolution using tensor products

  def initialise(self, *args):
    strideLayer.initialise(self, *args)
    self.setMode()

  def setMode(self, _conv_mode = None):
    if _conv_mode is not None: 
      self.conv_mode = _conv_mode
    elif self.conv_mode is None:
      self.conv_mode = CONVMODE_DEFAULT

  def setBatchSize(self, _batch_size = 1): # this assigns convolution mode
    """ 
    This class overloads this function and establishes actual convolution mode.
    """
    self.batch_size = int(_batch_size) if isint(_batch_size) else len(_batch_size)
    if not(self.batch_size): return

    # Pooling mode defaults to ranger
    
    self.mode = CONVMODE_FFT
    if np.all(self.strides == self.kernel):
      if self.conv_mode != CONVMODE_FFT:
        self.mode = CONVMODE_RANGER

    if self.mode is CONVMODE_RANGER:
      self.stride = strider(np.hstack((self.batch_size, self.input_maps, self.dims)),
                            np.hstack((self.maps, self.kernel)),
                            np.hstack(self.strides))

  def forward(self, _input_data = []):
    if self.batch_size != len(_input_data):
      self.setBatchSize(_input_data)
    if self.batch_size:
      self.forward_call[self.mode](_input_data)

#-------------------------------------------------------------------------------
