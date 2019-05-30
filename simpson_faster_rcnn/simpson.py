import csv
import os
import os.path

import torch.utils.data as data
from PIL import Image


def find_classes(dir):
    fname = os.path.join(dir, 'annotation.txt')
    # read the content of the file
    with open(fname) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    # find the list of classes
    classes = dict()
    for x in content:
        classes[x.split(",")[-1]] = 0

    # assign a label for each class
    index = 0
    for key in sorted(classes):
        classes[key] = index
        index += 1

    return classes


def make_dataset(dir, classes, set):
    images = []

    fname = os.path.join(dir, '%s_annotation.txt' % set)

    # read the content of the file
    with open(fname) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    for x in content:
        path = x.split(",")[0]
        label = classes[x.split(",")[-1]]
        item = (path, label)
        images.append(item)

    return images


def write_csv_file(dir, images, set):
    csv_file = os.path.join(dir, set + '.csv')
    if not os.path.exists(csv_file):

        # write a csv file
        print('[dataset] write file %s' % csv_file)
        with open(csv_file, 'w') as csvfile:
            fieldnames = ['name', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for x in images:
                writer.writerow({'name': x[0], 'label': x[1]})

        csvfile.close()


class SimpsonDataset(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.path_images = os.path.join(self.root, 'characters')

        self.classes = find_classes(self.root)
        self.images = make_dataset(self.root, self.classes, set)

        print('[dataset] Simpson set=%s  number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

        write_csv_file(self.root, self.images, set)

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
