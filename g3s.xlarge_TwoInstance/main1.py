import os
import json

tf_config = {
    'cluster': {
        'worker': ['172.31.18.137:4000', '172.31.28.184:4001']
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

num_workers = len(tf_config['cluster']['worker'])

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

multi_worker_model.fit(multi_worker_dataset, epochs=150, batch_size = global_batch_size)

later = time.time()
difference = later - now
print("\nTraining time: {}\n".format(difference))

save_path = "SavedModel/keyPointModel"
multi_worker_model.save(save_path)

#Most recently used instance type
results_dir = "Results/"

plt.plot(multi_worker_model.history.history['mae'])
plt.title('Training: Mean Absolute Error')
plt.ylabel('mae')
plt.xlabel('# epochs')
plt.savefig(results_dir + 'mae.png', bbox_inches='tight')
plt.clf()
plt.plot(multi_worker_model.history.history['accuracy'])
plt.title('Training: Accuracy')
plt.ylabel('acc')
plt.xlabel('# epochs')
plt.savefig(results_dir + 'acc.png', bbox_inches='tight')