#!/usr/bin/env bash

python split_dataset.py
python dataset.py
python train.py

python train_frcnn.py -p ./datasets/train_annotation.txt
python test_frcnn.py -p ./datasets/test_annotation.txt

python train_wildcat.py