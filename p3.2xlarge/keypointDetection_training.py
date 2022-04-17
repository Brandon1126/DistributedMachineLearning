import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from keras.layers import Convolution2D, BatchNormalization, Flatten, Dense, Dropout, MaxPool2D
from keras.models import Sequential

# os.add_dll_directory("/home/ubuntu/cuda/include")

now = time.time()

Train_Dir = '../training.csv'
train_data = pd.read_csv(Train_Dir)  

# Some labels are missing (28 in total)
# I've decided to fill these null values in instead of dropping them.
# This does affect accuracy, but not as much as dropping the images entirely (I think)
# Overall it has a negligible effect, but it has to be adjusted to prevent errors.
train_data.fillna(method = 'ffill',inplace = True)

# Extracting images one by one and placing them into "train_images"
train_images = []
for i in range(0,7049):
    temp_img = train_data['Image'][i].split(' ')
    temp_img = ['0' if x == '' else x for x in temp_img]
    train_images.append(temp_img)


# Converting training and testing images to np.array, and reshaping to be 96 by 96 by 1
train_images = np.array(train_images,dtype = 'float').reshape(-1,96,96,1)


# Now that the images have been set up, the image column can be dropped
# Will now extract the labels
# These labels are where the keypoints are located
training = train_data.drop('Image',axis = 1)

# Each label conststs of 15 (x, y) coordinates (30 values)
# These labels tell the model where the actual keypoints are
# Without these, the model couldn't "learn"
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
"""

Detailed Description:
I made this model up. It's a somewhat randomly constructed CNN model meant to handle
images inputs. The basic idea is that we wanted a model that had sufficient parameters to stress test
the GPUs, but also give decently accuracy when trained. This model ends up overfitting the data a bit.
But it does well overall.

Basic idea of CNN models:
The early convolution layers are meant to detect basic features, like edges, lines, etc.
The input layer is 96 by 96 to fit our data pixel by pixel. 
The padding='same' means that zeros will be filled in at the edges as our filter
'convols' around the edges. Output is then the same as the input size. Model breaks without it.
Batchnormalization helps to smooth out the training.

"""
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

"""
Compile Description:
The optimizer used is 'adam', which is a known algorithm for minimization of our loss function.
This optimizer controls how the model 'learns' and handles the mathematical details of how
the parameters of our model will change in order to make better predictions. There are many
different types of optimizers. I picked adam randomly.
Our loss function is 'mean_squared_error', there are many different loss functions that we could use
This loss function is what we're trying to minimize. A larger loss means that the models predicted
keypoint locations were further away from the actual locations. Our optimizer is tasked with
strategically minimizing this loss function.
The metrics we want to track is acurracy and mean absolute error.
Acurracy measures how well the model is doing, 100% would mean the model is perfect.
(Not likely to happen). mean absolute error is related to mean squared error (which is our loss function)
It has a lower value, makes it look nicer in our graph, that is the only reason I used it.
Lower is better.

"""
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae','accuracy'])


# Measuring how much time passed for preprocessing the data and building + compiling
# our CNN model.
later = time.time()
difference = later - now
print("\nInitialization time: {}\n".format(difference))
now = time.time()

"""

This line of code starts the training processing (model.fit)
Epochs = 100 means that we want to go through our data 100 times.
batch_size = 64 is how many images we want to send into our GPU at a time.
Higher batch_size means our model will train faster, but could have problems with
converging; meaning that the accuracy could bounce around more while the model is trying 
to learn.
validation_split=0.1 means that we want to use 10% of our data to validate how well the model is doing.
However, this means we have less overall training data, which is a trade off that we will make!

"""
model.fit(train_images,train_labels,epochs = 100,
          batch_size=64, validation_split=0.1)



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
plt.plot(model.history.history['val_accuracy'])
plt.title('Train Accuracy vs Validation Accuracy')
plt.ylabel('acc')
plt.xlabel('# epochs')
plt.savefig(results_dir + 'acc.png', bbox_inches='tight')


loss, mae, acc, *z = model.evaluate(train_images, train_labels, verbose=2)
print("Model accuracy with training data: {:5.2f}%".format(100 * acc))





