import inspect
import os
import sys
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from simpson_faster_rcnn.dataset import load_dataset

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
                  3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
                  7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
                  11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
                  14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

pic_size = 224
batch_size = 32
epochs = 200
num_classes = len(map_characters)


def create_model_four_conv(input_shape):
	"""
	CNN Keras model with 4 convolutions.
	:param input_shape: input shape, generally X_train.shape[1:]
	:return: Keras model, RMS prop optimizer
	"""
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
	return model, opt


def create_model_six_conv(input_shape):
	"""
	CNN Keras model with 6 convolutions.
	:param input_shape: input shape, generally X_train.shape[1:]
	:return: Keras model, RMS prop optimizer
	"""
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	return model, opt


def get_resnet(input_shape):
	return keras.applications.resnet50.ResNet50(input_shape=input_shape,
	                                            classes=num_classes, weights=None), SGD(lr=0.01, decay=1e-6,
	                                                                                    momentum=0.9, nesterov=True)


def load_model_from_checkpoint(weights_path, six_conv=False, input_shape=(pic_size, pic_size, 3)):
	if six_conv:
		model, opt = create_model_six_conv(input_shape)
	else:
		model, opt = create_model_four_conv(input_shape)
	model.load_weights(weights_path)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def lr_schedule(epoch):
	lr = 0.01
	return lr * (0.1 ** int(epoch / 10))


def training(model, X_train, X_test, y_train, y_test, data_augmentation=True):
	filepath = "./models/weights_6conv_%s.hdf5" % time.strftime("%Y%m%d")
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
	es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=15)

	csv_logger = CSVLogger('./logs/classification.log')

	callbacks_list = [LearningRateScheduler(lr_schedule), checkpoint, es, csv_logger]

	if data_augmentation:
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images
		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(X_train)
		history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
		                              steps_per_epoch=X_train.shape[0] // batch_size, epochs=40,
		                              validation_data=(X_test, y_test), callbacks=callbacks_list)
	else:
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
		                    shuffle=True, callbacks=callbacks_list)
	return model, history


if __name__ == '__main__':
	X_train, y_train = load_dataset('train')
	X_test, y_test = load_dataset('test')
	model, opt = get_resnet(X_train.shape[1:])
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model, history = training(model, X_train, X_test, y_train, y_test, data_augmentation=False)
