"""
# Note that all derivative functions must accept second inputs arguments that
# may contain the result of the transfer function if it not used.

"""
import numpy as np

#-------------------------------------------------------------------------------
def quadcf(y, a): # average cost function
  return 0.5 * np.mean( (y-a)**2 )

#-------------------------------------------------------------------------------
def quadcd(e, z, AD = None, a = None):
  if AD is None: return e
  if type(AD) is np.ndarray: return e * AD
  return e * AD(z, a)

#-------------------------------------------------------------------------------
def xentcf(y, a):
  loga = np.log(a + 1e-300)
  logb = np.log( (1. - a) + 1e-300)
  return -0.5 * np.mean(y * loga + (1-y)* logb)

#-------------------------------------------------------------------------------
def xentcd(e, *args): 
  return e

#-------------------------------------------------------------------------------
COST_FUNCTION_DERIVATIVE = {'quad': (quadcf, quadcd),
                            'xent': (xentcf, xentcd)}
#-------------------------------------------------------------------------------


