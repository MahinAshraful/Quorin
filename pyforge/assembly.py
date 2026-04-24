"""Numba-compiled assembly kernels. Implemented in Step 5.

Isolated in its own module so ``import pyforge`` does not pay Numba's
compilation cost unless an assemble path is actually called.
"""
