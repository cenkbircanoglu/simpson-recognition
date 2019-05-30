#!/usr/bin/env bash

python split_dataset.py
python dataset.py
python train.py

python train_frcnn.py -p ./datasets/train_annotation.txt

for i in ./datasets/test/characters/* ; do
    python test_frcnn.py -p  "$i"
done

python train_wildcat.py