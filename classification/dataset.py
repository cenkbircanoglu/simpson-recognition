import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = None


def _parse_fn(filename, label):
	global IMAGE_SIZE
	image_string = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string)
	image_normalized = (tf.cast(image_decoded, tf.float32) / 127.5) - 1
	image_resized = tf.image.resize(image_normalized, (IMAGE_SIZE, IMAGE_SIZE))
	return image_resized, label


def load_data(path='./../data/all.csv', batch_size=32, img_size=32):
	global IMAGE_SIZE
	IMAGE_SIZE = img_size
	image_csv = pd.read_csv(path)

	# Prepend image filenames in train/ with relative path
	filenames = image_csv['path'].tolist()
	labels = image_csv['label'].tolist()

	train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames,
	                                                                            labels,
	                                                                            train_size=0.8,
	                                                                            random_state=420)

	num_train = len(train_filenames)
	output_size = len(set(train_labels + val_labels))

	train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels))).map(
		_parse_fn).shuffle(buffer_size=10000).batch(batch_size)

	val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels))).map(
		_parse_fn).batch(batch_size)

	return train_data, val_data, num_train, output_size
