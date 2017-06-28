Issues:

08/04/2017: Speed peformance ~2X poorer than expected for given core count and CPU specification.  This is to be
resolved before introducing further functionality (e.g. different transfer/activation functions, cost functions,
momentum, dropout, regularisation etc...)

13/04/2017: NumPy is slow at matrix multiplication with OpenBLAS. Using MKL (via conda) increased CPU usage to multiple
cores without notable speedup - it often takes longer.

Todo:

Identify NumPy bottlenecks and resolve. This will probably require Cython with C++/assembler.
Nnet development is suspended until new C++ library CLAS is usable and Cython-wrapped.

