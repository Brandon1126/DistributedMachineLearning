import os
import json

os.environ['TF_CONFIG'] = '{"cluster": {"worker": ["3.136.84.118:12345", "3.143.223.239:23456"]}, "task": {"type": "worker", "index": 1} }'

import time
import tensorflow as tf
import keypoint_setup
import matplotlib as plt

now = time.time()

per_worker_batch_size = 128

#tf_config = json.loads(os.environ['TF_CONFIG'])
#num_workers = len(tf_config['cluster']['worker'])

num_workers = 2
strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = keypoint_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = keypoint_setup.build_and_compile_cnn_model()

later = time.time()
difference = later - now
print("\nInitialization time: {}\n".format(difference))
now = time.time()

multi_worker_model.fit(multi_worker_dataset, epochs=150, validation_split = 0.1, steps_per_epoch=50)

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


