# A module for windowing and windowed functions useful for convolution, pooling, and other such operations including
# indexing C-style row major arrays in their ravelled flat forms.

# Gary Bhumbra

import numpy as np
from numpy.fft import (rfftn, irfftn)
from scipy.fftpack import (fftn, ifftn)
from numpy.lib.stride_tricks import as_strided
from scipy.signal.signaltools import _centered


#-------------------------------------------------------------------------------
# Deal with multiple versions of NumPy and SciPy in various states of obsolences
#-------------------------------------------------------------------------------
# FFTPACK.next_regular
try:
  from scipy.signal.signaltools import _next_regular
except ImportError:
  from scipy.fftpack.helper import next_fast_len as _next_regular
#-------------------------------------------------------------------------------
# np.flip
def NPFLIP(X, a):
  Xslice = [slice(None)] * X.ndim
  Xslice[a] = slice(None, None, -1)
  return X[Xslice]

try:
  npflip = np.flip
except AttributeError:
  npflip = NPFLIP

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def slices(_I = None, _J = None, _D = None, N = None):
  """
  Returns a tuple of slices S where:

  S[0] = slice(I[0], J[0], D[0])
  ...
  S[n-1] = slice(I[n-1), J[n-1], D[n-1])

  `None' values are the default and single values
  are propagated if necessary. Note this function
  follows the convention of slice(J) beling 
  interpreted of slice(None, J).
  
  Optional input N can  be used to pre-append slices 
  with slice(None) entries to give an tuple of length N.
  """

  if (_I is not None and _J is None) and (_J is None and _N is None):
    I, J, D = np.atleast_1d(_J), np.atleast_1d(_I), np.atleast_1d(_D)
  else:
    I, J, D = np.atleast_1d(_I), np.atleast_1d(_J), np.atleast_1d(_D)

  IJD = np.array([len(I), len(J), len(D)], dtype = int)
  n = np.max(IJD)

  if N is None: N = n
  if np.any(n != IJD):
    if IJD[0] == 1: I = np.tile(I, n)
    if IJD[1] == 1: J = np.tile(J, n)
    if IJD[2] == 1: D = np.tile(D, n)
    IJD = np.array([len(I), len(J), len(D)], dtype = int)
    if np.any(n != IJD):
      raise ValueError("Input dimensions commensurate.")

  S = [slice(None)] * N
  d = N - n
  for i in range(n):
    S[i+d] = slice(I[i], J[i], D[i])

  return tuple(S)
  
#-------------------------------------------------------------------------------
def subarray(X, _I = [], _J = [], _D = None):
  '''
  Returns subarray of X from initial index vector _I to final index vector _J
  and optional interval specification _D.
  '''
  N = X.ndim
  I, J = np.atleast_1d(_I), np.atleast_1d(_J)
  
  if _D is None: # For performance, separate each of these

    IJ = np.array([len(I), len(J)], dtype = int)
    n = np.max(IJ)

    if np.any(n != IJ):
      if IJ[0] == 1: I = np.tile(I, n)
      if IJ[1] == 1: J = np.tile(J, n)
      IJ = np.array([len(I), len(J)], dtype = int)
      if np.any(n != IJ):
        raise ValueError("Input dimensions commensurate.")

    # Deal with unitary step trivial cases first

    if N == 1:
      return X[I[0]:J[0]]
    elif N == 2:
      if n == 1: return X[:, I[0]:J[0]]
      return X[I[0]:J[0], I[1]:J[1]]
    elif N == 3:
      if n == 2: return X[:, I[0]:J[0], I[1]:J[1]]
      if n == 1: return X[:, :, I[0]:J[0]]
      return X[I[0]:J[0], I[1]:J[1], I[2]:J[2]]
    else:
      D = np.tile(_D, n)

  else:

    D = np.atleast_1d(_D)
    IJD = np.array([len(I), len(J), len(D)], dtype = int)
    n = np.max(IJD)

    if np.any(n != IJD):
      if IJD[0] == 1: I = np.tile(I, n)
      if IJD[1] == 1: J = np.tile(J, n)
      if IJD[2] == 1: D = np.tile(D, n)
      IJD = np.array([len(I), len(J), len(D)], dtype = int)
      if np.any(n != IJD):
        raise ValueError("Input dimensions commensurate.")

    # Now deal with non-unitary step trivial cases

    if n != len(D):
      raise ValueError("Initial and interval specification argument dimensionality incommensurate")
    if N == 1:
      return X[I[0]:J[0]:D[0]]
    elif N == 2:
      if n == 1: return X[:, I[0]:J[0]:D[0]]
      return X[I[0]:J[0]:D[0], I[1]:J[1]:D[1]]
    elif N == 3:
      if n == 2: return X[:, I[0]:J[0]:D[0], I[1]:J[1]:D[1]]
      if n == 1: return X[:, :, I[0]:J[0]:D[0]]
      return X[I[0]:J[0]:D[0], I[1]:J[1]:D[1], I[2]:J[2]:D[2]]

  # Use slices for all non-trivial solutions

  I = np.hstack( (np.zeros(N-len(I), dtype = int), I) ) 
  J = np.hstack( (X.shape[:(N-len(J))], J) )
  D = np.hstack( (np.ones(N-len(D), dtype = int), D) ) 

  return X[slices(I, J, D)]

#-------------------------------------------------------------------------------
def reverse(X, _ax = None): 
  '''
  Similar to numpy.flip but with two changes:
  # by default flips all axes
  # attempts views to attempt fast execution for common 1D, 2D, and 3D flips
  '''
  N = np.ndim(X)
  ax = np.arange(N) if _ax is None else np.unique(_ax)
  
  # Deal with quick view flips first
  if N == len(ax):
    if N == 1: return X[::-1]
    if N == 2: return X[::-1, ::-1]
    if N == 3: return X[::-1, ::-1, ::-1]
    return X.ravel()[::-1].reshape(X.shape)

  n = len(ax)
  if not(n): return X
  if N == n + 1 and ax[0]:
    if N == 2: return X[:, ::-1]
    if N == 3: return X[:, ::-1, ::-1]

  # Otherwise resort to slices 

  s = [slice(None)] * N
  for _ax in ax:
    s[_ax] = slice(None, None, -1)
  return X[s]

#-------------------------------------------------------------------------------
def subsample(X, _d):
  '''
  Returns sub-sampled values of X according to the interval specification d. 
  '''
  d = np.atleast_1d(_d)
  N = np.ndim(X)
  n = len(d)

  # Deal with no subsampling 

  if not(n) or np.all(d == 1): 
    return X
  elif n > N: 
    raise ValueError("Subsampling specification exceeds input dimensionality.")

  # Deal with trivial low dimension cases

  if N == len(d):
    if N == 1: return X[::d[0]]
    if N == 2: return X[::d[0], ::d[1]]
    if N == 3: return X[::d[0], ::d[1]]
  n = len(d)
  if not(n): return X

  if N == n + 1:
    if N == 2: return X[:, ::d[0]]
    if N == 3: return X[:, ::d[1], ::d[1]]
  
  # Deal with other cases - note there is no need to pre-append d
  s = [slice(None)] * N
  s[-n:] = [slice(None, None, d_) for d_ in d]
  return X[s]

#-------------------------------------------------------------------------------
def intrapad(X, _d, x = 0.):  # this is trickier
  """
  Interleaves values x (default 0.) within array X according axis specification d.
  """
  N = np.ndim(X)
  d = np.atleast_1d(_d)
  n = len(d)

  if not(n) or np.all(d == 1): 
    return X
  elif n > N: 
    raise ValueError("Interleave specification exceeds input dimensionality.")

  # Deal with trivial low-dimension solutions.
  if N < 4:
    S = np.atleast_1d(X.shape)
    S[N-n:] *= d
    Y = np.tile(x, S)

    if N == 1:
      Y[::d[0]] = X
      return Y
    elif N == 2:
      if n == 1:
        Y[:, ::d[0]] = X
      else:
        Y[::d[0], ::d[1]] = X
      return Y
    else: # N = 3
      if n == 1:
        Y[:, :, ::d[0]] = X
      elif n == 2:
        Y[:, ::d[0], ::d[1]] = X
      else:
        Y[::d[0], ::d[1], ::d[2]] = X
      return Y

  # Loop required for higher dimensionality
  d = np.atleast_1d(np.hstack((np.zeros(N-len(d), dtype = int), d)))
  Y = None

  for i in range(N):
    di = d[i]
    if di > 1:
      if Y is None: Y = X
      S = np.atleast_1d(Y.shape)
      S[i] *= di
      Y = np.expand_dims(Y, i+1)
      s = np.atleast_1d(Y.shape)
      y = np.tile(x, s)
      Y = np.stack([Y] + [y]*(di-1),  axis = i+1).reshape(S)

  return Y
 
#-------------------------------------------------------------------------------
# Function of convenience for convolution/pooling operations.

def strider(_X, _E = [], _S = []): 
  """
  Outputs `fancy-indexed' array XI (if first input floating point) or the 
  `fancy-indexing' array I itself (if first input integer) where: 
    X = input array (D = X.shape if X is floating point otherwise D = X)
    E = window dimensions
    S = overlap dimensions

  - fastest read-usage of I is XI = np.take(X, I)
  - fastest write usage of I is: XA = np.zeros(X.size, dtype = float); np.add.at(XA, I, DX)
  """

  # First deal with inputs and pre-append appropriate defaults for _E and _S
  
  X, E, S = np.atleast_1d(_X), np.atleast_1d(_E), np.atleast_1d(_S)
  isfp = X.dtype is np.dtype('float64')
  D = X.shape if isfp else X
  n, e, s = len(D), len(E), len(S)
  _e, _s = n - e, n - s

  if _e < 0:
    raise ValueError("Window specification dimensionality input dimensionality.")
  elif _e:
    E = np.hstack( (np.ones(_e), E) )
  if _s < 0:
    raise ValueError("Stride specification dimensionality input dimensions.")
  elif _s:
    S = np.hstack( (E[:_s], S) )

  if np.any(E>D):
    raise ValueError("Window dimension(s) exceeds input dimension(s).")
  if np.any(S>E):
    raise ValueError("Overlap dimension(s) exceeds window dimension(s).")

  e = np.minimum(_e, _s)

  # Calculate cumulative products of dimensions
  C = np.hstack((1, np.cumprod(D)))
  R = C[-1]//C
  c = int(np.prod(E))

  # Calculate number of strides
  B = D - E # `border' dimensions
  if np.any(np.mod(B, S)):
    raise ValueError("(Input.dims-window.dims) indivisible by stride dimension(s).")
  N = np.ones(n, dtype = int) + (B / S)
  r = int(np.prod(N))

  # Calculate maxuend from excerpt dimensions but independent of stride
  M = np.empty((1, c), dtype = int) 
  m = np.zeros(n, dtype = int)
  for j in range(c):
    k = n - 1
    if j: m[k] += 1
    done = m[k] < E[k]
    while not(done):
      m[k] = 0
      k -= 1
      done = k < 0
      if not(done):
        m[k] += 1
        done = m[k] < E[k]
    M[0, j] = np.inner(R[1:], m)
     
  # Calculate addend from stride but independent of excerpt dimensions
  A = np.empty((r, 1), dtype = int) 
  a = np.zeros(n, dtype = int)
  for i in range(r):
    k = n - 1
    if i: a[k] += S[k]
    done = a[k] <= B[k]
    while not(done):
      a[k] = 0
      k -= 1
      done = k < 0
      if not(done):
        a[k] += S[k]
        done = a[k] <= B[k]
    A[i, 0] = np.inner(R[1:], a)

  # Sum = addend + maxuend
  I = A + M


  # Redimensionalise I if necessary
  if e:
    De = D[:e]
    I = np.reshape(I, np.hstack((De, int(r/np.prod(De)), c)))

  # Output 
  if isfp: return np.take(X, I)
  return I

#-------------------------------------------------------------------------------
def conv(in1, in2, mode = None, In1 = None, In2 = None, **_kwds):
  """A version of scipy.signal.fftconvolve modified in the following respects:
  
    - convolution of scalars, integers, or empty arrays is not supported.
    - thread-safety for execution using Numpy prior to version 1.9 not checked.
    - in1 and in2 are entirely commutative including for `valid' mode.
    - if either input is an integer array it is treated as dimension specification
      and the Fourier transform of the other input is outputted.
    - if both inputs are integer arrays, they are treated as dimension specifications
      and the dimensions of the corresponding Fourier dimensions are outputted as a
      tuple of shapes in the following order (In1.shape, In2.shape, Out.shape).
    - optional inputs In1 and In2 may be entered as the Fourier transforms if pre-
      computed for performance - the size of the Fourier domain is adjusted
      accordingly. If either of optional inputs In1 or In2 are integer array, they 
      specify the input dimensions of the Fourier domain; the output dimensions will be
      adjusted accordingly.
    - _kwds allows specification of axes (e.g. axes = [0,1]) to convolve, defaulting 
      to all axes of the input with fewer dimensions - note unconvolved dimensions
      must be either commensurate singleton for one input for valid broadcasting.

    Note that `mode' is ignored unless both in1 and in2 are floating point arrays.

  """

  if mode is None: mode = 'full'

  # Check input data types and shapes and homogenise dimensionality if required
  in1,   in2   = np.asarray(in1), np.asarray(in2)
  in1dt, in2dt = in1.dtype,       in2.dtype;

  sh1 = np.atleast_1d(in1) if in1dt is np.dtype('int64') else np.atleast_1d(in1.shape)
  sh2 = np.atleast_1d(in2) if in2dt is np.dtype('int64') else np.atleast_1d(in2.shape)

  mind, maxd = np.minimum(len(sh1), len(sh2)), np.maximum(len(sh1), len(sh2))
  difd = maxd - mind

  if difd:
    if len(sh1) < maxd:
      sh1 = np.hstack( (np.ones(difd, dtype = int), sh1) )
      if in1dt is not np.dtype('int64'): in1 = in1.reshape(sh1)
    else:
      sh2 = np.hstack( (np.ones(difd, dtype = int), sh2) )
      if in2dt is not np.dtype('int64'): in2 = in2.reshape(sh2)

  s1, s2 = np.maximum(sh1, sh2), np.minimum(sh1, sh2) 

  # Default axes or confirm axes specification gives compatible dimensions in unconvolved axes
  kwds = dict(_kwds)
  if 'axes' not in kwds:
    ax = np.arange(mind) + difd 
  else:
    ax = np.unique(np.atleast_1d(kwds['axes']))
    notax = np.ones(maxd, dtype = bool)
    notax[ax] = False
    sh1na, sh2na = sh1[notax], sh2[notax]
    if np.any(np.logical_and(sh1na != sh2na, np.logical_and(sh1na !=1, sh2na !=1))):
      raise ValueError("Unconvolved dimensions incommensurate.")

  shape = s1[ax] + s2[ax] - 1
  fshape = [_next_regular(int(sz)) for sz in shape]

  # Check dimensions of Fourier inputs if entered
  Fshape, Sh1, Sh2 = None, None, None
  if In1 is not None: 
    In1 = np.asarray(In1)
    Sh1 = np.atleast_1d(In1) if In1.dtype is np.dtype('int64') else np.atleast_1d(In1.shape) 
    if not Sh1.ndim == sh1.ndim:
        raise ValueError("Input dimensionality incommensurate")
    Fshape = Sh1[ax]
  if In2 is not None: 
    In2 = np.asarray(In2)
    Sh2 = np.atleast_1d(In2) if In2.dtype is np.dtype('int64') else np.atleast_1d(In2.shape) 
    if not Sh2.ndim == sh2.ndim:
        raise ValueError("Input dimensionality incommensurate")
    if Sh1 is not None:
      if not Sh1.ndim == Sh2.ndim:
        raise ValueError("Fourier input dimensionality incommensurate")
      if not np.all(Sh1[ax] == Sh2[ax]):
        raise ValueError("Fourier input dimensions incommensurate within domain axes")
      notax = np.ones(maxd, dtype = bool)
      notax[ax] = False
      Sh1na, Sh2na = Sh1[notax], Sh2[notax]
      if np.any(np.logical_and(Sh1na != Sh2na, np.logical_and(Sh1na !=1, Sh2na !=1))):
        raise ValueError("Fourier input dimensions incommensurate outside domain axes")
    Fshape = Sh2[ax]

  # Adjust Fourer input or output dimensions if required
  if Fshape is None:
    Fshape = np.copy(fshape)
    Fshape[-1] = Fshape[-1] // 2 + 1
  else:
    _fshape = np.copy(Fshape)
    _fshape[-1] = (_fshape[-1] -1) * 2
    if np.any(_fshape != fshape) and fshape[-1] != _fshape[-1] + 1:
      fshape = _fshape

  # Output sizes or Fourier transforms according to input data types
  if in1dt is np.dtype('int64'):
    if in2dt is np.dtype('int64'): 
      in1s, in2s, outs = np.copy(sh1), np.copy(sh2), np.copy(s1)
      in1s[ax], in2s[ax], out[ax] = fshape, fshape, Fshape
      return in1s, in2s, outs
    else:
      In2 = fftn(in2, fshape, axes = ax) if np.issubdtype(in2dt, np.complex) else rfftn(in2, fshape, axes = ax)
      return In2
  elif in2dt is np.dtype('int64'):
    In1 = fftn(in1, fshape, axes = ax) if np.issubdtype(in1dt, np.complex) else rfftn(in1, fshape, axes = ax)
    return In1

  # Prepare slices for Fourier axes used for convolution
  fslice = [slice(None)] * len(s1)
  for i in range(len(ax)):
    fslice[ax[i]] = slice(None, shape[i])
  fslice = tuple(fslice)

  # Apply convolution theorem
  if np.issubdtype(in1dt, np.complex) or np.issubdtype(in2dt, np.complex):
    if In1 is None: In1 = fftn(in1, fshape, axes = ax)
    if In2 is None: In2 = fftn(in2, fshape, axes = ax)
    out = ifftn(In1 * In2, fshape, axes = ax)[fslice].copy()
  else:
    if In1 is None: In1 = rfftn(in1, fshape, axes = ax)
    if In2 is None: In2 = rfftn(in2, fshape, axes = ax)
    out = irfftn(In1 * In2, fshape, axes = ax)[fslice].copy()

  # Return array or subarray according to mode
  if mode == "full":
      return out

  censh = np.atleast_1d(out.shape)
  if mode == "same":
    censh[ax] = s1[ax]
  elif mode == "valid":
    censh[ax] = s1[ax] - s2[ax] + 1
  else:
      raise ValueError("Acceptable mode flags are 'valid',"
                       " 'same', or 'full'.")

  return _centered(out, censh)

#-------------------------------------------------------------------------------
def poolaxes(X, _E): 
  '''
  Changes dimensions and axes for non-overlapping pooling in last dimension where:
  X = input
  E = pooling window dimensions
  '''
  E = np.atleast_1d(_E)
  n = len(_E)

  # Fast-track trivial solutions 
  if n == X.ndim:
    if n == 1: 
      return X.reshape(X.shape[0]/E[0], E[0])
    elif n == 2: 
      Q = X.shape[0]//E[0], X.shape[1]//E[1]
      return X.reshape(Q[0], E[0], Q[1], E[1]).swapaxes(1,2).reshape(Q[0]*Q[1], E[0]*E[1])

  N = X.ndim
  p = N - n
  if p < 0:
    raise ValueError("Input dimensionality incommensurate")

  O = np.array(X.shape)
  P, D = O[:p], O[p:]
  Q = D // E

  # Deal with simple cases first (n < 3)
  if n == 1:
    return X.reshape(np.hstack( (P, Q[0], E[0]) ))
  if n == 2: 
    preshape = np.hstack( (P, Q[0], E[0], Q[1], E[1]) )
    return X.reshape(preshape).swapaxes(1+p, 2+p).reshape(np.hstack((P, np.prod(Q), np.prod(E))))

  # Deal with multidimensional cases
  I = np.arange(N-3+p, p, -2, dtype = int)
  J = np.arange(N-2+p, n-1+p, -1, dtype = int)

  x = X.reshape( np.hstack( (P, np.reshape(Q, (n, 1)), np.reshape(E, (n, 1))) ).ravel() )

  for k in range(n-1):
    x = x.swapaxes(I[k], J[k])

  return x.reshape( np.hstack((P, np.prod(Q), np.prod(E))) )

#-------------------------------------------------------------------------------
def unpoolaxes(X, _E, _O): 
  ''' 
  Inverse of poolaxes - ouput dimensions (input O) must be given to recover 
  pre-pool shape where:

  X = input (i.e. output of poolaxes)
  E = pooling window dimensions
  O = final output dimensions
  '''

  if X.ndim != 2:
    raise ValueError("First input must be two-dimensional")
  E, O = np.atleast_1d(_E), np.atleast_1d(_O)
  n = len(E)

  # Fast-track trivial solutions 
  if n == len(O):
    if n == 1: 
      return X.reshape(_O)
    elif n == 2: 
      Q = O[0]//E[0], O[1]//E[1]
      return X.reshape(Q[0], Q[0], E[1], E[1]).swapaxes(1,2).reshape(O)

  N = len(O)
  p = N - n
  if p < 0:
    raise ValueError("Input dimensionality incommensurate")

  P, D = O[:p], O[p:]
  Q = D // E

  # Deal with simple cases first (n < 3)
  if n == 1:
    return X.reshape(O)
  if n == 2: 
    preshape = np.hstack( (P, Q[0], Q[0], E[1], E[1]) )
    return X.reshape(preshape).swapaxes(1+p, 2+p).reshape(O)

  # Deal with multidimensional cases
  I = np.arange(N-3+p, p, -2, dtype = int)[::-1]
  J = np.arange(N-2+p, n-1+p, -1, dtype = int)[::-1]

  x = X.reshape( np.hstack( (P.ravel(), Q.ravel(), E.ravel()) ) )

  for k in range(n-1):
    x = x.swapaxes(I[k], J[k])

  return x.reshape( np.hstack((P, np.prod(Q), np.prod(E))) )

#-------------------------------------------------------------------------------

