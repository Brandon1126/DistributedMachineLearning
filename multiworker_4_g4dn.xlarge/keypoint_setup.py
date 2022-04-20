import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Convolution2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout
from keras.models import Sequential


# This function preprocesses the data based on given batch_size
# Batch_size can change based on the number of instance worker nodes
def keypoint_dataset(i, j):
    print("Made it to keypoint_dataset")
    train_dir = '../training.csv'
    train_data = pd.read_csv(train_dir)
    train_data.fillna(method='ffill', inplace=True)
    train_images = []
    for i in range(i, j):
        temp_img = train_data['Image'][i].split(' ')
        temp_img = ['0' if x == '' else x for x in temp_img]
        train_images.append(temp_img)

    train_images = np.array(train_images, dtype='float').reshape(-1, 96, 96, 1) / 255.0
    training = train_data.drop('Image', axis=1)

    train_labels = []
    for i in range(i, j):
        temp_label = training.iloc[i, :]
        train_labels.append(temp_label)

    train_labels = np.array(train_labels, dtype='float')

    return train_images, train_labels


# This function builds a custom-made Convolutional neural network
# and return the model to the caller
def build_and_compile_cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1), padding='same'))
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(96, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(30))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae', 'accuracy'])

    print("made it to compiled model")

    return model
