import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, MaxPooling2D, Softmax, Dropout, Activation
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as k


class SmallerVGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        #Depth(Channels) last
        model = Sequential()
        inputshape = (height, width, depth)
        chanDim = -1


        #Depth(Channels) first
        if k.image_data_format() == 'channels_first':
            inputshape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputshape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
