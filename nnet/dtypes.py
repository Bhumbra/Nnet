# A data types module with functions to improve support for duck-typing.

# Gary Bhumbra

import numpy as np

#-------------------------------------------------------------------------------
def Type(x):
  y = type(x)
  if y is np.float32:
    return float
  elif y is np.float64:
    return float
  elif y is np.int32:
    return int  
  elif y is np.int64:
    return int  
  elif y is np.bool_:
    return bool
  else:
    return y
#-------------------------------------------------------------------------------
def isarray(x):
  y = type(x)
  if y is list:
    return True
  elif y is tuple:
    return True
  elif y is np.ndarray:
    return True
  elif y is np.matrixlib.defmatrix.matrix:
    return True
  else:
    return False

#-------------------------------------------------------------------------------
def isbool(x):
  if Type(x) is bool: return True
  return False

#-------------------------------------------------------------------------------
def isint(x):
  if Type(x) is int: return True
  return False

#-------------------------------------------------------------------------------
def isfloat(x):
  if Type(x) is float: return True
  return False

#-------------------------------------------------------------------------------
def isnum(x):
  if isbool(x): return True
  if isint(x): return True
  if isfloat(x): return True
  return False

#-------------------------------------------------------------------------------

