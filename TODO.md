Completed:

baseLayer, feedLayer
CostLayer
strideLayer, PoolLayer, ConvLayer

Issues:

08/04/2017: Speed peformance ~2X poorer than expected for given core count and CPU specification.
This must be resolved before introducing further functionality (e.g. different transfer/activation functions,
cost functions, momentum, dropout, regularisation etc...)

13/04/2017: Numpy seems slow at matrix multiplication with OpenBLAS. Rather tragically, using MKL (via conda) increased
CPU usage to multiple cores without any speedup - actually it took longer.

Todo:

Identify numpy bottlenecks and resolve. This will probably require ctypes with C++/inline assembler.


