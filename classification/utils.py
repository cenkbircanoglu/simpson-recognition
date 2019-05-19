import matplotlib.pyplot as plt


def draw_losses(history, name):
	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	plt.figure(figsize=(8, 8))
	plt.subplot(2, 1, 1)
	plt.plot(acc, label='Training Accuracy')
	plt.plot(val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.ylabel('Accuracy')
	plt.title('Training and Validation Accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.ylabel('Cross Entropy')
	plt.title('Training and Validation Loss')
	plt.xlabel('epoch')
	plt.savefig('./plots/%s' % name)
