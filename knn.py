import importlib
import pandas as pd 
import numpy as np 
import sklearn
utils = importlib.import_module('utils')

x_train, y_train = utils.read_train_data("train.csv")
x_pred = utils.read_test_data("test.csv")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()