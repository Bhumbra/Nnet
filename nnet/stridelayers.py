"""
A module for layer classes for strided operations with feature maps. These can be used for supervised
and unsupervsed learning. For classical convolutional networks, the classes ConvLayer and PoolLayer
will be most useful here.

Here I use `kernel' to refer to the weight coefficient or pooling size, and `filter' to be the actual 
convolution filter (i.e. rotated weight_coef).
"""

# Gary Bhumbra.

from nnet.baselayers import *

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

  def_stride = 1 # default stride - which can be adjusted by other classes (0 matches coef_size)
  single_map = None
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
    if self.strides is None: self.strides = np.tile(self.def_strides, len(self.dims))

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

