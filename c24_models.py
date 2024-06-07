#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.arange(5); y_true = x ** 2
plt.plot(x, y_true, "ro")

def f1(x, a, b): return a * x + b
def f2(x, a, b, c): return a * b ** x + c
models = [(f1, "b--"), (f2, "g--")]

for f, style in models:
    popt, _ = curve_fit(f, x, y_true)
    y_pred = f(x, *popt)  # args unpacking
    mse = ((y_true - y_pred) ** 2).mean()
    print(style, mse, popt)
    plt.plot(x, y_pred, style)

plt.show()
