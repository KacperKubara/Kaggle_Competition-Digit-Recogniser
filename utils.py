import numpy as np 
import pandas as pd 

def read_train_data(path):
    x = pd.read_csv(path).drop(['label'], axis = 1)
    y = pd.read_csv(path)['label']
    return x, y
def read_test_data(path):
    x = pd.read_csv(path)
    return x