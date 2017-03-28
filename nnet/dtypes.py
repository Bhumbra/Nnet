# A data types module

# Gary Bhumbra

import numpy as np

#-------------------------------------------------------------------------------
def Type(x):
  y = type(x)
  if y == np.float64:
    return float
  elif y == np.int64:
    return int  
  elif y == np.bool_:
    return bool
  else:
    return y
#-------------------------------------------------------------------------------
def isarray(x):
  if type(x) is list:
    return True
  if type(x) is tuple:
    return True
  if type(x) is np.ndarray:
    return True
  if type(x) is np.matrixlib.defmatrix.matrix:
    return True
  return False

#-------------------------------------------------------------------------------
def isbool(x):
  if type(x) is bool or type(x) is np.bool_: return True
  return False

#-------------------------------------------------------------------------------
def isint(x):
  if type(x) is int: return True
  if type(x) is np.int32 or type(x) is np.int64: return True
  return False

#-------------------------------------------------------------------------------
def isfloat(x):
  if type(x) is float: return True
  if type(x) is np.float32 or type(x) is np.float64: return True
  return False

#-------------------------------------------------------------------------------
def isnum(x):
  if isbool(x): return True
  if isint(x): return True
  if isfloat(x): return True
  return False

#-------------------------------------------------------------------------------
def argtrue(x):
  i = np.nonzero(x)
  if len(i) == 1: 
    i = i[0]
    if len(i): return i
    return ()
  return i

#-------------------------------------------------------------------------------

