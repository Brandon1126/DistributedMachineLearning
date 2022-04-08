import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import os
import time

# os.add_dll_directory("/home/ubuntu/cuda/include")


import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, MaxPool2D, ZeroPadding2D

now = time.time()

Train_Dir = '../training.csv'
train_data = pd.read_csv(Train_Dir)  

#dealing with missing values, I've decided to fill null values in instead of dropping them
train_data.fillna(method = 'ffill',inplace = True)

train_images = []
for i in range(0,7049):
    temp_img = train_data['Image'][i].split(' ')
    temp_img = ['0' if x == '' else x for x in temp_img]
    train_images.append(temp_img)


#converting training and testing images to np.array, and reshaping to be 96 by 96 by 1
train_images = np.array(train_images,dtype = 'float').reshape(-1,96,96,1)


#Now that the images have been set up, the image column can be dropped
training = train_data.drop('Image',axis = 1)


train_labels = []
for i in range(0,7049):
    temp_label = training.iloc[i,:]
    train_labels.append(temp_label)


train_labels = np.array(train_labels,dtype = 'float')

# Testing to make sure the shapes fit
print(train_images.shape)
print(train_labels.shape)

#Model Creation
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
model.add(Dense(30))

model.summary()

model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae','accuracy'])

later = time.time()
difference = later - now
print("\nInitialization time: {}\n".format(difference))
now = time.time()


model.fit(train_images,train_labels,epochs = 100,
          batch_size = 64)

later = time.time()
difference = later - now
print("\nTraining time: {}\n".format(difference))

save_path = "SavedModel/keyPointModel"
model.save(save_path)

#Most recently used instance type
results_dir = "Results/"

plt.plot(model.history.history['mae'])
plt.title('Mean Absolute Error')
plt.ylabel('mae')
plt.xlabel('# epochs')
plt.savefig(results_dir + 'mae.png', bbox_inches='tight')
plt.clf()
plt.plot(model.history.history['accuracy'])
plt.title('Training Accuracy')
plt.ylabel('acc')
plt.xlabel('# epochs')
plt.savefig(results_dir + 'acc.png', bbox_inches='tight')


loss, mae, acc, *z = model.evaluate(train_images, train_labels, verbose=2)
print("Model accuracy with training data: {:5.2f}%".format(100 * acc))





