import numpy as np 
import pandas as pd 
import cv2
def read_train_data(path):
    x = pd.read_csv(path).drop(['label'], axis = 1)
    y = pd.read_csv(path)['label']
    return x, y
def read_test_data(path):
    x = pd.read_csv(path)
    return x
def save_data(path, data):
    data = pd.DataFrame(data, columns = ["Label"])
    data.index.name = "ImageId"
    data.index += 1
    data.to_csv(path)
# Deskew the image
def deskew():
    im_array = pd.read_csv("train.csv", nrows= 10)
    img = im_array.drop(['label'], axis = 1).iloc[0,:].values
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    print(img)
    return img