import os
import tensorflow as tf
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Conv2D, MaxPool2D, Dropout

# This function preprocesses the data based on given batch_size
# Batch_size can change based on the number of instance worker nodes
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

    #Key line to partition the data based on batch_size, 
    #The data will be split between the VMs
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).batch(batch_size)
    print("Made it to return keypoint_dataset")
    return train_dataset

# This function builds a custom-made Convolutional neural network
# and return the model to the caller
def build_and_compile_cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(96,96,1), padding='same'))
    model.add(Convolution2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
    model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(96, (3,3), activation='relu', padding='same'))
    model.add(Convolution2D(96, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(30))
    
    model.summary()

    model.compile(optimizer='adam', 
                loss='mean_squared_error',
                metrics=['mae','accuracy'])

    print("made it to compiled model")

    return model

