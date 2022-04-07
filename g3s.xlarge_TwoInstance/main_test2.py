import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = {
    'cluster': {
        'worker': ['localhost:4000', 'localhost:4001']
    },
    'task': {'type': 'worker', 'index': 1}
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

