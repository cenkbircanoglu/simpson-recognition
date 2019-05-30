import errno
import inspect
import os
import shutil
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def copy_file(src, dest):
    try:
        shutil.copy(src, dest)
    except IOError as e:
        # ENOENT(2): file does not exist, raised also on missing dest parent dir
        if e.errno != errno.ENOENT:
            raise
        # try creating parent directories
        os.makedirs(os.path.dirname(dest))
        shutil.copy(src, dest)


def create_train_test(path):
    df = pd.read_csv(path, header=None, names=['path', 'x1', 'x2', 'x3', 'x4', 'class'])
    stratified_split = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    for train_idx, test_idx in stratified_split.split(df, df['class']):
        train_annotations = df.iloc[train_idx]
        train_annotations['new_path'] = train_annotations['path'].apply(
            lambda x: x.replace('datasets', 'datasets/train'))
        test_annotations = df.iloc[test_idx]
        test_annotations['new_path'] = test_annotations['path'].apply(lambda x: x.replace('datasets', 'datasets/test'))
        print(len(df), len(train_annotations), len(test_annotations))
        for i, item in train_annotations.iterrows():
            copy_file(item['path'], item['new_path'])
        for i, item in test_annotations.iterrows():
            copy_file(item['path'], item['new_path'])
        train_annotations[['new_path', 'x1', 'x2', 'x3', 'x4', 'class']].to_csv('./datasets/train_annotation.txt',
                                                                                header=None, index=False)
        test_annotations[['new_path', 'x1', 'x2', 'x3', 'x4', 'class']].to_csv('./datasets/test_annotation.txt',
                                                                               header=None, index=False)


if __name__ == '__main__':
    create_train_test('./datasets/annotation.txt')
