import tensorflow as tf


class Config:
	model_name = None
	img_size = 224
	batch_size = 32


class DenseNet121Config(Config):
	model_name = 'densenet121'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.densenet.DenseNet121(input_shape=(cls.img_size, cls.img_size, 3),
		                                                  include_top=False, weights='imagenet')


class DenseNet169Config(Config):
	model_name = 'densenet169'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.densenet.DenseNet169(input_shape=(cls.img_size, cls.img_size, 3),
		                                                  include_top=False, weights='imagenet')


class DenseNet201Config(Config):
	model_name = 'densenet201'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.densenet.DenseNet201(input_shape=(cls.img_size, cls.img_size, 3),
		                                                  include_top=False, weights='imagenet')


class InceptionResNetV2Config(Config):
	model_name = 'inceptionresnetv2'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(cls.img_size, cls.img_size, 3),
		                                                                   include_top=False, weights='imagenet')


class InceptionV3Config(Config):
	model_name = 'inceptionv3'
	img_size = 299
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.inception_v3.InceptionV3(input_shape=(cls.img_size, cls.img_size, 3),
		                                                      include_top=False, weights='imagenet')


class MobileNetConfig(Config):
	model_name = 'mobilenet'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.mobilenet.MobileNet(input_shape=(cls.img_size, cls.img_size, 3), include_top=False,
		                                                 weights='imagenet')


class MobileNetV2Config(Config):
	model_name = 'mobilenetv2'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(cls.img_size, cls.img_size, 3),
		                                                      include_top=False, weights='imagenet')


class NASNetMobileConfig(Config):
	model_name = 'nasnetmobile'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.nasnet.NASNetMobile(input_shape=(cls.img_size, cls.img_size, 3),
		                                                 include_top=False, weights='imagenet')


class NASNetLargeConfig(Config):
	model_name = 'nasnetlarge'
	img_size = 331
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.nasnet.NASNetLarge(input_shape=(cls.img_size, cls.img_size, 3),
		                                                include_top=False, weights='imagenet')


class ResNet50Config(Config):
	model_name = 'resnet50'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.resnet50.ResNet50(input_shape=(cls.img_size, cls.img_size, 3),
		                                               include_top=False, weights='imagenet')



class VGG16Config(Config):
	model_name = 'vgg16'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.vgg16.VGG16(input_shape=(cls.img_size, cls.img_size, 3),
		                                         include_top=False, weights='imagenet')


class VGG19Config(Config):
	model_name = 'vgg19'
	img_size = 224
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.vgg19.VGG19(input_shape=(cls.img_size, cls.img_size, 3),
		                                         include_top=False, weights='imagenet')


class XceptionConfig(Config):
	model_name = 'xception'
	img_size = 299
	batch_size = 32

	@classmethod
	def load_model(cls):
		return tf.keras.applications.Xception(input_shape=(cls.img_size, cls.img_size, 3),
		                                      include_top=False, weights='imagenet')


all_configs = [ResNet50Config,
               #DenseNet121Config, DenseNet169Config, DenseNet201Config, InceptionResNetV2Config, InceptionV3Config,
               #MobileNetConfig, MobileNetV2Config, NASNetLargeConfig, NASNetMobileConfig, ResNet50Config, VGG16Config,
               #VGG19Config, XceptionConfig
               ]
