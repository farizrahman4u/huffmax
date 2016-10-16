from huffmax import Huffmax
from keras.layers import Dense, Input
from keras.models import Sequential, Model
import numpy as np
import time


batch_size = 32
input_dim = 100
nb_classes = 100000
nb_samples = 10000

times = {}

X = np.random.random((nb_samples, input_dim))
Y_huffmax = np.random.randint(0, nb_classes, size=(nb_samples, 1))

Y_softmax = []
for _ in range(nb_samples):
	oh = np.zeros(nb_classes)
	oh[np.random.randint(0, nb_classes)] = 1
	Y_softmax += [oh]
Y_softmax = np.array(Y_softmax)


softmax_model = Sequential()
softmax_model.add(Dense(input_dim=input_dim, output_dim=nb_classes, activation='softmax'))
softmax_model.compile(loss='mse', optimizer='sgd')
softmax_model.predict(X[:1])

start_time = time.time()
softmax_model.fit(X, Y_softmax, batch_size=batch_size)
end_time = time.time()
times['Softmax'] = end_time - start_time

del softmax_model

for mode in [0, 1]:
	vector = Input((input_dim,))
	target_class = Input((1,))
	probability = Huffmax(nb_classes, verbose=True, mode=mode)([vector, target_class])
	huffmax_model = Model(input=[vector, target_class], output=probability)
	huffmax_model.compile(loss='mse', optimizer='sgd')
	huffmax_model.predict([X[:1], Y_huffmax[:1]])
	start_time = time.time()
	huffmax_model.fit([X, Y_huffmax], np.ones((nb_samples, 1)), batch_size=batch_size)
	end_time = time.time()
	times['Huffmax (mode ' + str(mode) + ')'] = end_time - start_time

for key in times.keys():
	print(key + ' : ' + str(times[key]))
