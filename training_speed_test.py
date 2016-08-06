from huffmax import Huffmax
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
import datetime


batch_size = 100000
input_dim = 100
nb_classes = 10000


times = {}

X = np.random.random((batch_size, input_dim))
Y_huffmax = np.random.randint(0, nb_classes, size=(batch_size, 1))
Y_softmax = []
for _ in range(batch_size):
	oh = np.zeros(nb_classes)
	oh[np.random.randint(0, nb_classes)] = 1
	Y_softmax += [oh]
Y_softmax = np.array(Y_softmax)

softmax_model = Sequential()
softmax_model.add(Dense(input_dim=input_dim, output_dim=nb_classes, activation='softmax'))
softmax_model.compile(loss='mse', optimizer='sgd')
softmax_model.predict(X[:1])

start_time = datetime.datetime.now()

softmax_model.fit(X, Y_softmax, batch_size=1000)

end_time = datetime.datetime.now()

times['Softmax'] = end_time - start_time

for mode in [0, 1]:
	vector = Input((input_dim,))
	target_class = Input((1,))
	probability = Huffmax(nb_classes, verbose=True, mode=mode)([vector, target_class])
	huffmax_model = Model(input=[vector, target_class], output=probability)
	huffmax_model.compile(loss='mse', optimizer='sgd')
	huffmax_model.predict([X[:1], Y_huffmax[:1]])

	start_time = datetime.datetime.now()

	huffmax_model.fit([X, Y_huffmax], np.ones((batch_size, 1)), batch_size=1000)

	end_time = datetime.datetime.now()

	times['Huffmax (mode ' + str(mode) + ')'] = end_time - start_time


for key in times.keys():
	print(key + ' : ' + str(times[key]))

