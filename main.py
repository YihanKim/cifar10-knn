
# Ytr : 50000 x 1 int (0 .. 9)

import numpy as np

# data reader

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

Xtr_rows = []
Ytr = []

for i in range(1, 6):
	data = unpickle("data/data_batch_%s" % str(i))
	Xtr_rows += list(data[b'data'])
	Ytr += list(data[b'labels'])

Xtr_rows = np.asarray(Xtr_rows)
Ytr = np.asarray(Ytr)

class NearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, y):
		self.Xtr = X
		self.ytr = y

	def predict(self, X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		for i in range(num_test):
			distances = np.sum(np.square(self.Xtr - X[i, :]), axis = 1)
			min_index = np.argmin(distances)
			Ypred[i] = self.ytr[min_index]
			print("%s / %s" % (i, num_test))
		return Ypred

cifar_nn = NearestNeighbor()

cifar_nn.train(Xtr_rows[:49900], Ytr[:49900])
Ypred = cifar_nn.predict(Xtr_rows[49900:])

equal = [1 if Ypred[i] == Ytr[49900 + i] else 0 for i in range(100)]

print(sum(equal) / 100)

