#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import numpy as np

### Strides

a = np.arange(12, dtype=np.int64).reshape((3, 4))
print(a.shape, a.strides)

b = a.reshape(4, 3)  # new view, different shape
print(b)
print(b.strides)
print(b.ravel())  # flat view

print(a.T)  # transposed view, same as a.transpose()
print(a.T.shape, a.T.strides)
print(a.T.ravel())
print(a.T.ravel("K"))  # real order of data in memory!

### Row-major 

a = np.array([[1, 2], [3, 4]], order="C", dtype="i1")
print(a)  # array([[1, 2], [3, 4]], dtype=int8)
print(a.strides)  # (2, 1)
print(a.ravel("K"))  # array([1, 2, 3, 4], dtype=int8)

### Column-major

a = np.array([[1, 2], [3, 4]], order="F", dtype="i1")
print(a)  # array([[1, 2], [3, 4]], dtype=int8)
print(a.strides)  # (1, 2)
print(a.ravel("K"))  # array([1, 3, 2, 4], dtype=int8)

### Changing data type

a = np.array([4.25, -5.75], dtype=">f2")
print(a)
print(a.data.tobytes().hex(" "))
a.dtype = "i1"
print(a)
print(a.data.tobytes().hex(" "))