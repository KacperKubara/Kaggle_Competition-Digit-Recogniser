import importlib
import pandas as pd 
import numpy as np 
utils = importlib.import_module('utils')

# Load data in
x_train, y_train = utils.read_train_data("train.csv")
x_test = utils.read_test_data("test.csv")

# SVC classiffier with polynomial kernel of 4th degree 
from sklearn import svm
svc = svm.SVC(kernel = 'poly', degree = 4)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

# Save the data
save_data("results.csv", y_pred)