Completed

baseLayer
feedLayer
CostLayer
strideLayer
PoolLayer
ConvLayer

Issues

08/04/2017: Speed peformance 2.5X poorer than expected for given core count and CPU specification.
This must be resolved before introducing further functionality (e.g. momentum, dropout, regularisation etc...)

Causes
This is most likely not the fault of numpy but an openBLAS bottleneck gemm executing np.multiple and np.einsum
What we do know is:
- Matrix multiplication in openBLAS numpy is very slow, at least 2X-3X as it should be.
- In purely mathematical terms, FFT-based convolution should be faster than summed products.

Todo

Optimise numpy-dependent code AMAP and confirm openBLAS is cause.
Possible solutions: 1) try MKL 2) rewrite gemm/fft using inline assembler using my C++ old classes (must convert 32-bit
to 64 bit!)



