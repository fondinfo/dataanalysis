#!/usr/bin/env python3
"""
@author  Michele Tomaiuolo - http://www.ce.unipr.it/people/tomamic
@license This software is free - https://opensource.org/license/mit
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
    ConfusionMatrixDisplay as CMD)
from sklearn import datasets, svm

digits = datasets.load_digits()

# Split data into a training set (70%) and a test set (30%)
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, shuffle=True)

# Learn the digits on the train subset
clf = svm.SVC()
clf.fit(x_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(x_test)

print(classification_report(y_test, predicted))

print(CMD.from_predictions(y_test, predicted).confusion_matrix)
