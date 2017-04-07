# Transfer functions and derivatives
# Note that all derivative functions must accept second inputs arguments that
# may contain the result of the transfer function if it not used.

# Gary Bhumbra

import numpy as np

#-------------------------------------------------------------------------------
def sigval(x, expmin = -69.):
  """
  y = np.zeros(x.shape, dtype = float)
  ok = x > expmin
  y[ok] = 1./(1.+np.exp(-x[ok]))
  return y
  """
  return 0.5 * np.tanh(0.5*x) + 0.5

#-------------------------------------------------------------------------------
def sigder(x, _sx = None):
  sx = _sx if _sx is not None else sigval(x)
  return sx * (1. - sx)

#-------------------------------------------------------------------------------
def ReLU(x):
  y = np.array(x, dtype = float)
  y[x<0.] = 0.
  return y

#-------------------------------------------------------------------------------
def ReDU(x, _sx = None):
  y = np.ones(x.shape, dtype = float)
  y[x<0.] = 0.
  return y

#-------------------------------------------------------------------------------
TRANSFER_FUNCTION_DERIVATIVE = {'none': (None,   None),
                                'sigm': (sigval, sigder),
                                'relu': (ReLU, ReDU)}
#-------------------------------------------------------------------------------
