import glob

import cv2
import h5py
import keras
import numpy as np

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
                  3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
                  7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
                  11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
                  14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

pic_size = 64
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15


def load_pictures(BGR, name):
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob('./datasets/%s/characters/%s/*' % (name, char))]
        nb_pic = round(pictures_per_class / (1 - test_size)) if round(pictures_per_class / (1 - test_size)) < len(
            pictures) else len(pictures)
        # nb_pic = len(pictures)
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size, pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels)


def create_dataset(BGR=False, name='train'):
    X, y = load_pictures(BGR, name)
    y = keras.utils.to_categorical(y, num_classes)
    h5f = h5py.File('./datasets/%s.h5' % name, 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('y', data=y)
    h5f.close()


def load_dataset(data_type='train'):
    h5f = h5py.File('./datasets/%s.h5' % data_type, 'r')
    X = h5f['X'][:].astype('float32') / 255.
    y = h5f['y'][:]
    h5f.close()

    print("Data ", X.shape, y.shape)

    return X, y


if __name__ == '__main__':
    create_dataset(name='train')
    create_dataset(name='test')
    load_dataset('train')
    load_dataset('test')
