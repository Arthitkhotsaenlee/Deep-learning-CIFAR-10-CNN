import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import sys
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.python.keras.callbacks import ModelCheckpoint

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
    input1 = Input(shape=input_shape)
    ## block 1 ##
    block1 = Conv2D(32,(3,3),activation="relu", kernel_initializer="he_uniform", padding="same",input_shape=input_shape)(input1)
    block1 = Conv2D(32,(3,3),activation="relu", kernel_initializer="he_uniform", padding="same",input_shape=input_shape)(block1)
    block1 = MaxPooling2D(2,2)(block1)
    block1 = Dropout(0.2)(block1)
    ### block2 ###
    block2 = Conv2D(64,(3,3),activation="relu", kernel_initializer="he_uniform", padding="same",input_shape=input_shape)(block1)
    block2 = Conv2D(64,(3,3),activation="relu", kernel_initializer="he_uniform", padding="same",input_shape=input_shape)(block2)
    block2 = MaxPooling2D(2, 2)(block2)
    block2 = Dropout(0.2)(block2)
    ## block3 ##
    block3 = Conv2D(128,(3,3),activation="relu", kernel_initializer="he_uniform", padding="same",input_shape=input_shape)(block2)
    block3 = Conv2D(128,(3,3),activation="relu", kernel_initializer="he_uniform", padding="same",input_shape=input_shape)(block3)
    block3 = MaxPooling2D(2, 2)(block3)
    block3 = Dropout(0.2)(block3)
    ## flatten layer ##
    flatten = Flatten()(block3)
    ## last layer ##
    last_ly = Dense(128,activation="relu", kernel_initializer="he_uniform")(flatten)
    last_ly = Dropout(0.2)(last_ly)
    last_ly = Dense(64, activation="relu", kernel_initializer="he_uniform")(last_ly)
    last_ly = Dropout(0.2)(last_ly)
    last_ly = Dense(32, activation="relu", kernel_initializer="he_uniform")(last_ly)
    last_ly = Dropout(0.2)(last_ly)
    output1 = Dense(10, activation="softmax", kernel_initializer="he_uniform")(last_ly)

    return Model(inputs=input1,outputs=output1)


def summerize_model(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()



if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()
    # get model
    model = CNN_model()
    # compile models
    # Model check point
    model_path = "CIFAR-10.h5"
    checkpoint = ModelCheckpoint(filepath= model_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')
    # fit model
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=32,
                        epochs=500,
                        validation_split=0.25,
                        callbacks=[checkpoint],
                        validation_data=(X_test,y_test),
                        verbose=True)
    _, acc = model.evaluate(X_test,y_test, verbose=False)
    print(">%.3f"%(acc*100))
    summerize_model(history)
    print("finish")
