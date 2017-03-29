"""
A module for layer classes for strided operations with feature maps. These can be used for supervised
and unsupervsed learning. For classical convolutional networks, the classes ConvLayer and PoolLayer
will be most useful here.
"""

# Gary Bhumbra.

from nnet.baselayers import *

class strideLayer(BackLayer):
  """
  This is a BackLayer with explicit support for feature maps each containing arbitrary dimensions. No
  multidimensional operations are supported in forward() or backward() because this is camel-cased
  base class intended to be inheriting to other classes, notable ConvLayer and PoolLayer and never
  intended to be instanstated itself.
  """

  def initialise(self, *args):
    BackLayer.initialise(self, *args)

