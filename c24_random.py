#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import numpy as np
rng = np.random.default_rng()

print(rng.random(size=(2, 3)))  # uniform in [0.0, 1.0)
print(rng.normal(size=(2, 3)))  # normal Gaussian, mean=0, stddev=1

stdev, mean = 5, 2.5
a = stdev * rng.normal(size=(2, 3)) + mean
b = rng.normal(mean, stdev, size=(2, 3))

print(rng.choice(["one", "two"], (2, 3)))

print(rng.integers(0, 10, (2, 3)))  # rng.choice(range(10), (2, 3))

a = np.arange(6)
rng.shuffle(a)  # modifies the array itself
print(a)
