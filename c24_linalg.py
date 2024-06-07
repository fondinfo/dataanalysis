#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import numpy as np

### Product

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
print(np.linalg.det(a))
b = np.linalg.inv(a)
print(b)
print(a @ b)
print(np.eye(2))  # unit 2x2 matrix; "eye" represents "I"

### Fair tickets

a = np.array([[1., 1.],
              [5., 3.]])
b = np.array([2150., 8350.])
print(np.linalg.inv(a) @ b)  # b: 1D, as column vector
print(np.linalg.solve(a, b))

### Eigenvalues, eigenvectors

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
w, v = np.linalg.eig(a)
print(w)  # eigenvalues
print(np.linalg.det(w[0] * np.eye(2) - a))  # 0.0
print(v)  # eigenvectors: normalized, in columns (!️)
print(np.sum(v**2, axis=0))  # magnitude = 1
print(a @ v.T[0] - w[0] * v.T[0])  # array([0, 0])

