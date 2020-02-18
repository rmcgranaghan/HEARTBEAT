def dynamic_range_and_mae_loss(y_true, y_pred):
	'''
	This is a loss function that contains mean absolute error and dynamic range
	terms. It requires the keras library and uses tensorflow notation. It can be
	added to a keras model at the 'compile' stage such as: model.compile(loss=dynamic_range_and_mae_loss)
	'''

	import keras as k
	mae = k.losses.mean_absolute_error(y_true, y_pred)
	dynamic_range_true = k.backend.max(y_true) - k.backend.min(y_true)
	dynamic_range_pred = k.backend.max(y_pred) - k.backend.min(y_pred)
	dynamic_range_loss = dynamic_range_true - dynamic_range_pred
	dynamic_range_loss = k.backend.mean(dynamic_range_loss)
	return mae + 0.1* dynamic_range_loss
