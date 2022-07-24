import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm


def load_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # convert X fron int to float
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    # normalization x
    X_train = X_train/255
    X_test = X_test/255
    #one-hot encoding
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    return X_train, y_train, X_test, y_test



def one_hot(y):
    y = np.reshape(y,-1)
    one_hot_y = np.zeros((y.shape[0],y.max()+1))
    one_hot_y[np.arange(y.shape[0]),y] = 1
    return one_hot_y

def CNN_model(input_shape=(32,32,3)):
    input = input_shape



if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()
    print("finish")
