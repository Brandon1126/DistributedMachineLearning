import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os
import random


from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, MaxPool2D, ZeroPadding2D
import tensorflow as tf


Test_Dir = '../training.csv'
test_data = pd.read_csv(Test_Dir)  

#dealing with missing values, I've decided to fill null values in instead of dropping them
#This, unfortunately, will affect accuracy, but I think it's better than dropping them
test_data.fillna(method = 'ffill',inplace = True)

test_images = []
for i in range(0,7049):
    temp_img = test_data['Image'][i].split(' ')
    temp_img = ['0' if x == '' else x for x in temp_img]
    test_images.append(temp_img)


#converting training and testing images to np.array, and reshaping to be 96 by 96 by 1
train_images = np.array(test_images,dtype = 'float').reshape(-1,96,96,1)


#Now that the images have been set up, the image column can be dropped
testing = test_data.drop('Image',axis = 1)

test_labels = []
for i in range(0,7049):
    temp_label = testing.iloc[i,:]
    test_labels.append(temp_label)


test_labels = np.array(test_labels,dtype = 'float')

#converting training and testing images to np.array, and reshaping to be 96 by 96 by 1
test_images = np.array(test_images,dtype = 'float').reshape(-1,96,96,1)


#Recovering the trained model
save_path = "SavedModel/keyPointModel"
model = tf.keras.models.load_model(save_path)



# These will be the random indexs of images to print
# This way I can see how the predictions across several randomly selected images performed
random_numbers = []
for i in range(20):
    random_numbers.append(random.randint(0, len(test_images) - 1))

# Each of these hold 5 sets of locations
predicted_image_features = []
predictions = model.predict(test_images)

for number in random_numbers:
    predicted_image_features.append(predictions[number])


plt.figure(figsize=(10, 10))
for i in range(10):
    fig1 = plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    fig2 = plt.subplot(4, 5, i + 11)
    plt.xticks([])
    plt.yticks([])

    for j in range(0, 30, 2):
      fig1.plot(predicted_image_features[i][j],predicted_image_features[i][j+1],'ro', markersize = 5)
      fig1.imshow(test_images[random_numbers[i]].reshape(96, 96), cmap="gray")
      fig2.plot(test_labels[i][j],test_labels[i][j+1],'bo', markersize = 5)
      fig2.imshow(test_images[random_numbers[i]].reshape(96, 96), cmap="gray")



results_dir = "Results/"

plt.subplots_adjust(wspace=10, hspace=0)
plt.tight_layout()
plt.savefig(results_dir + 'predictions.png', bbox_inches='tight')
#plt.show()
print('Predictions done')
