#!/bin/bash

echo "Training Model"
python3 keypointDetection_training.py |& tee Results/g4dn.xlarge.log
echo "Done with training"
python3 keypointDetection_predictions.py
echo "Done with predictions"
