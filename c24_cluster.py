#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
rng = np.random.default_rng()

px = rng.normal(size=1000) * 10
py = np.concatenate([rng.normal(size=500),
                     rng.normal(size=500) + 5])
data = np.stack((px, py), axis=1)
min_, max_ = data.min(axis=0), data.max(axis=0)
data_norm = (data - min_) / (max_ - min_)

kmeans = cluster.KMeans(n_clusters=2)
labels = kmeans.fit_predict(data)
labels_norm = kmeans.fit_predict(data_norm)

plt1 = plt.subplot(311)  # rows, cols, index
plt1.scatter(px, py)
plt1.title.set_text("Original dataset")

plt2 = plt.subplot(312)  # rows, cols, index
plt2.scatter(px, py, c=labels)
plt2.title.set_text("Clustering w/o normalization")

plt3 = plt.subplot(313)  # rows, cols, index
plt3.scatter(px, py, c=labels_norm)
plt3.title.set_text("Clustering with normalization")

plt.subplots_adjust(hspace=1)
plt.show()
