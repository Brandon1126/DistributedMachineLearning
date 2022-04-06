#!/bin/bash

echo "Training Model"
python3 keypointDetection_training.py | tee Results/training_output.txt
echo "Done with training"
python3 keypointDetection_predictions.py | tee Results/prediction_output.txt
echo "Done with predictions"