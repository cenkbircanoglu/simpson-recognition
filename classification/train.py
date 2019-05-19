import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from classification.config import all_configs
from classification.dataset import load_data
from classification.utils import draw_losses

print(tf.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

num_epochs = 50
val_steps = 20
learning_rate = 0.0001
lr_finetune = learning_rate / 10
fine_tune_epochs = 50


def train(config):
	batch_size = config.batch_size
	img_size = config.img_size
	train_data, val_data, num_train, output_size = load_data(batch_size=batch_size, img_size=img_size)
	base_model = config.load_model()
	model_name = config.model_name
	base_model.trainable = False

	maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
	prediction_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')

	model = tf.keras.Sequential([
		base_model,
		maxpool_layer,
		prediction_layer
	])
	es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
	mc = ModelCheckpoint('./models/weights_%s.h5' % model_name, monitor='val_acc', mode='max', verbose=1,
	                     save_best_only=True)

	csv_logger = CSVLogger('./logs/%s.log' % model_name)
	model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

	print(model.summary())
	steps_per_epoch = round(num_train) // batch_size
	history = model.fit(train_data.repeat(), epochs=num_epochs, steps_per_epoch=steps_per_epoch,
	                    validation_data=val_data.repeat(), validation_steps=val_steps, callbacks=[es, mc, csv_logger])
	model.save_weights('./models/weights_%s.h5' % model_name)
	draw_losses(history, model_name)

	def finetuning():
		base_model.trainable = True

		for layer in base_model.layers[:100]:
			layer.trainable = False

		model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_finetune), metrics=['acc'])
		print(model.summary())

		total_epochs = num_epochs + fine_tune_epochs
		csv_logger = CSVLogger('./logs/finetuned_%s.log' % model_name)
		mc = ModelCheckpoint('./models/weights_finetuned_%s.h5' % model_name, monitor='val_acc', mode='max', verbose=1,
		                     save_best_only=True)

		history_fine = model.fit(train_data.repeat(), steps_per_epoch=steps_per_epoch, epochs=total_epochs,
		                         initial_epoch=num_epochs, validation_data=val_data.repeat(), validation_steps=val_steps,
		                         callbacks=[es, mc, csv_logger])
		# Save fine-tuned model weights
		model.save_weights('./models/weights_finetuned_%s.h5' % model_name)
		draw_losses(history_fine, 'finetuned_%s' % model_name)

	finetuning()


if __name__ == '__main__':

	for config in all_configs:
		try:
			train(config)
		except Exception as e:
			print(e, config.model_name)
