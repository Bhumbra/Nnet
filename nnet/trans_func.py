# Transfer functions and derivatives
# Note _all_ transfer functions and derivatives _must_ accept keyword arguments
# and handle the output keyword argument out=z correctly.

# Gary Bhumbra

import numpy as np
import scipy.special

#-------------------------------------------------------------------------------
"""
def sigval(x, **kwds):
  # return 1./(1+exp(-x))
  # return 0.5 * np.tanh(0.5*x) + 0.5

  z = kwds["out"] if "out" in kwds else np.empty_like(x)
  np.multiply(x, 0.5, out=z)
  np.tanh(z, out=z)
  np.multiply(z, 0.5, out=z)
  np.add(z, 0.5, out=z)
  return z
"""
sigval = scipy.special.expit

#-------------------------------------------------------------------------------
def sigder(x, **kwds):
  #y = sigval(x); return (1.-y)*y 
  z = kwds["out"] if "out" in kwds else np.empty_like(x)
  y = kwds["val"] if "val" in kwds else sigval(x)
  np.subtract(1., y, out=z)
  np.multiply(z, y, out=z)
  return z 

#-------------------------------------------------------------------------------
def ReLU(x, **kwds):
  z = kwds["out"] if "out" in kwds else np.empty_like(x)
  y = kwds["ind"] if "ind" in kwds else x < 0
  np.copyto(z, x, casting='no')
  z[y].fill(0.)
  return z

#-------------------------------------------------------------------------------
def ReDU(x, **kwds):
  z = kwds["out"] if "out" in kwds else np.empty_like(x)
  y = kwds["ind"] if "ind" in kwds else x < 0
  z.fill(1.)
  z[y].fill(0.)
  return z

#-------------------------------------------------------------------------------
TRANSFER_FUNCTION_DERIVATIVE = {'none': (None,   None),
                                'sigm': (sigval, sigder),
                                'relu': (ReLU, ReDU)}
#-------------------------------------------------------------------------------
