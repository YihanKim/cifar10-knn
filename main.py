
import math
import numpy as np

# import knn model from "model.py"
from model import KNearestNeighbor


### 1. Data reading

# Data truncation constant

#            DATA_TRAIN     TOTAL_READ
#                 |  DATA_USE    |
# 0               |     |        |
# --------------------------------
# |     TRAIN     | VAL |   ..   |
# --------------------------------

DATA_USE = 50000 			# max 50000
DATA_VALIDATE = 100			# should be greater than 0, less than DATA_USE
DATA_TRAIN = DATA_USE - DATA_VALIDATE

DATA_BATCH_SIZE = 10000		# unit batch size

# CIFAR-10 Data reader
# (from https://www.cs.toronto.edu/~kriz/cifar.html)

def unpickle(file):
	'''
	The archive contains the files data_batch_1, ..., data_batch_5, as well as test_batch. 
	Each of these files is a Python "pickled" object produced with cPickle.
	'''
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

# Each data batch has 10,000 images recorded as 3,072(= 32 * 32 * 3) integers.
# Read data as much as we need (<= 50000, determined by value of DATA_USE),
# and save it(data and labels) to the X_rows and Y respectively.

X_rows = []
Y = []

for i in range(1, 1 + math.ceil(DATA_USE / DATA_BATCH_SIZE)):
	data = unpickle("data/data_batch_%s" % str(i))
	X_rows += list(data[b'data'])
	Y += list(data[b'labels'])

X_rows = np.asarray(X_rows)	
Y = np.asarray(Y)

# Each row vector in X contains 3,072 integer belongs to 0 to 255.
# Y contains integer from 0 to 9, which indicates 10 classes in CIFAR-10.
# if Y[i] == Y[j] for i != j, 
# X[i] and X[j] represents the different object in same class(such as car).


### 2. Data splitting

# Now, split the data and the labels into training set and validation set.
# CIFAR-10 batch data has mixed order of class, so we don't need to shuffle it. 

Xtr_rows = X_rows[:DATA_TRAIN]
Ytr = Y[:DATA_TRAIN]

Xval_rows = X_rows[DATA_TRAIN:DATA_USE]
Yval = Y[DATA_TRAIN:DATA_USE]


### 3. Running k-NN model

# Use k-NN classifier for CIFAR-10

cifar_knn = KNearestNeighbor() # kNN model defined in "model.py"
cifar_knn.train(Xtr_rows, Ytr) # give training data into model
validation_accuracies = []

for k in [1, 3, 5, 10, 20, 50, 100]:
	Yval_predict = cifar_knn.predict(Xval_rows, k = k)
	acc = np.mean(Yval_predict == Yval)
	print ('accuracy for k = %s: %f' % (k, acc))
	# use a particular value of k and evaluation on validation data
	validation_accuracies.append((k, acc))

print("running results: ", end="")
print(validation_accuracies)

