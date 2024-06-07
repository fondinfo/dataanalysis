#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import numpy as np
import pandas as pd

df = pd.DataFrame({"A" : ["foo", "bar", "foo", "bar",
                          "foo", "bar", "foo", "foo"],
                   "B" : ["one", "one", "two", "three",
                          "two", "two", "one", "three"],
                   "C" : np.random.randn(8),
                   "D" : np.random.randn(8)})
print(df.groupby("A").sum())  # only numeric data
print()

df = pd.DataFrame({
    "A" : ["one", "one", "two", "three"] * 3,
    "B" : ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
    "C" : np.random.randn(12)})
print(pd.pivot_table(df, index="A", columns="B", values="C"))
