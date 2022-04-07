import os
import tensorflow as tf
import numpy as np
import pandas as pd

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, MaxPool2D, ZeroPadding2D

# This function preprocesses the data based on given batch_size
def keypoint_dataset(batch_size):
    print("Made it to keypoint_dataset")
    Train_Dir = '../training.csv'
    train_data = pd.read_csv(Train_Dir) 
    train_data.fillna(method = 'ffill',inplace = True)
    train_images = []
    for i in range(0,7049):
        temp_img = train_data['Image'][i].split(' ')
        temp_img = ['0' if x == '' else x for x in temp_img]
        train_images.append(temp_img)

    train_images = np.array(train_images,dtype = 'float').reshape(-1,96,96,1)
    training = train_data.drop('Image',axis = 1)

    train_labels = []
    for i in range(0,7049):
        temp_label = training.iloc[i,:]
        train_labels.append(temp_label)

    train_labels = np.array(train_labels,dtype = 'float')

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(60000).repeat().batch(batch_size)
    print("Made it to return keypoint_dataset")
    return train_dataset

def build_and_compile_cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    
    model.summary()

    model.compile(optimizer='adam', 
                loss='mean_squared_error',
                metrics=['mae','accuracy'])

    print("made it to compiled model")

    return model

