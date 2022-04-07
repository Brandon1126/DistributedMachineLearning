import os
import json

tf_config = {
    'cluster': {
        'worker': ['localhost:4000', 'localhost:4001']
    },
    'task': {'type': 'worker', 'index': 0}
}


os.environ['TF_CONFIG'] = json.dumps(tf_config)

print(os.environ['TF_CONFIG'])

import numpy as np
import random
import pandas as pd
import time
import tensorflow as tf
import keypoint_setup
import matplotlib as plt

now = time.time()

per_worker_batch_size = 128

#tf_config = json.loads(os.environ['TF_CONFIG'])
#num_workers = len(tf_config['cluster']['worker'])

num_workers = 2

print(os.environ['TF_CONFIG'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

print("Made it past strategy")

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = keypoint_setup.keypoint_dataset(global_batch_size)

print("Made it past data")

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = keypoint_setup.build_and_compile_cnn_model()

later = time.time()
difference = later - now
print("\nInitialization time: {}\n".format(difference))
now = time.time()

multi_worker_model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=50)

later = time.time()
difference = later - now
print("\nTraining time: {}\n".format(difference))

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




# These will be the random indexs of images to print
# This way I can see how the predictions across several randomly selected images performed
random_numbers = []
for i in range(10):
    random_numbers.append(random.randint(0, len(test_images) - 1))

# Each of these hold 5 sets of locations
predicted_image_features = []
predictions = multi_worker_model.predict(test_images)

for number in random_numbers:
    predicted_image_features.append(predictions[number])


plt.figure(figsize=(10, 10))
for i in range(10):
    fig1 = plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])

    for j in range(0, 30, 2):
      fig1.plot(predicted_image_features[i][j],predicted_image_features[i][j+1],'ro', markersize = 3)
      fig1.plot(test_labels[i][j],test_labels[i][j+1],'bo', markersize = 3)
      fig1.imshow(test_images[random_numbers[i]].reshape(96, 96), cmap="gray")



results_dir = "Results/"

plt.subplots_adjust(wspace=10, hspace=0)
plt.tight_layout()
plt.savefig(results_dir + 'predictions.png', bbox_inches='tight')
print('Predictions done')