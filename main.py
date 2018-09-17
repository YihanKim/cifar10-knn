
# Ytr : 50000 x 1 int (0 .. 9)

import numpy as np
from NearestNeighbor import NearestNeighbor

# constant

DATA_TOTAL = 10000 			# min 10000 ~ max 50000
DATA_VALIDATE = 200
DATA_TRAIN = DATA_TOTAL - DATA_VALIDATE

# data reader

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

Xtr_rows = []
Ytr = []

for i in range(1, 1 + DATA_TOTAL // 10000):
	data = unpickle("data/data_batch_%s" % str(i))
	Xtr_rows += list(data[b'data'])
	Ytr += list(data[b'labels'])

Xtr_rows = np.asarray(Xtr_rows)
Ytr = np.asarray(Ytr)

cifar_nn = NearestNeighbor()

cifar_nn.train(Xtr_rows[:DATA_TRAIN], Ytr[:DATA_TRAIN])
Ypred = cifar_nn.predict(Xtr_rows[DATA_TRAIN:])

equal = [1 if Ypred[i] == Ytr[DATA_TRAIN + i] else 0 for i in range(DATA_VALIDATE)]

print(sum(equal) / DATA_VALIDATE)

