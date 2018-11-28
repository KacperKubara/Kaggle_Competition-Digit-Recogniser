import importlib
import pandas as pd 
import numpy as np 
utils = importlib.import_module('utils')

# Load data in
x_train, y_train = utils.read_train_data("train.csv")
x_test = utils.read_test_data("test.csv")

# Split for the train and test dataset 
from sklearn.model_selection import train_test_split

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_jobs = -1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# Save the data 

 