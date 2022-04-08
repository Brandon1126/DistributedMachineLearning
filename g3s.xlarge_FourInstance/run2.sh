#!/bin/bash

echo "Training Model"
python3 main1.py
echo "Done with training"
python3 keypointDetection_predictions.py
