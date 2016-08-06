# Make sure that the huffmax layer outputs a valid probability distribution

from huffmax import Huffmax
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np


batch_size = 32
input_dim = 100
nb_classes = 10


x_vector = np.random.random((batch_size, input_dim))
x_classes = np.array([range(nb_classes)] * batch_size)


for mode in [0, 1]:
	input_vector = Input((input_dim,))
	target_classes = Input((nb_classes,))
	probabilies = Huffmax(nb_classes, verbose=True, mode=mode)([input_vector, target_classes])
	model = Model(input=[input_vector, target_classes], output=probabilies)
	model.compile(loss='mse', optimizer='sgd')
	y = model.predict([x_vector, x_classes])
	assert y.shape == (batch_size, nb_classes)
	assert np.all(y > 0)
	assert np.all(y < 1)
	assert np.sum(y) == batch_size
	print('mode ' + str(mode) + ' ok.')
