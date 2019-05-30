#!/usr/bin/env bash


wget https://pjreddie.com/media/files/yolov3.weights


python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5

python train.py

