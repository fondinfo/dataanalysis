#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Create a random dataset
X = 360 * np.random.rand(80, 1)
y = np.sin(np.radians(X.flatten()))
y[::4] += 1 - 2 * np.random.rand(20)

# Fit two regression models
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 360.0, 1).reshape(360, 1)
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot data and predictions
plt.figure()
plt.scatter(X, y, c="blue", label="data")
plt.plot(X_test, y_2, color="red", label="model2")
plt.plot(X_test, y_1, color="green", label="model1")
plt.legend()
plt.show()
