
import numpy as np

# K-Nearest Neighbor Class
# (from http://cs231n.github.io/classification/, modified slightly)

class KNearestNeighbor(object):
	'''
	K-nearest Neighbor class
	
	train : Saves training data(lists of data and its labels). That's it.
	predict : Consumes data list and parameter k, return list of predicted class number
	'''
	def __init__(self):
		# do nothing
		pass

	def train(self, X, y):
		""" 
		X is N x D where each row is an example. 
		Y is 1-dimension of size N 
		return None
		"""
		# the nearest neighbor classifier simply remembers all the training data
		self.Xtr = X
		self.ytr = y
		return

	def predict(self, X, k = 1):
		""" 
		X is N x D where each row is an example we wish to predict label for.
		k is hyperparameter; the number of neighbors that use to predict.
		return list of class indices with length N.
		"""
		num_test = X.shape[0]
		# lets make sure that the output type matches the input type
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		# loop over all test rows
		for i in range(num_test):
			# find the nearest training image to the i'th test image
			# using the L1 distance (sum of absolute value differences)
			# You can change distance function as you need.
			# -----------------------
			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
			# distances = np.sum(np.square(self.Xtr - X[i, :]), axis = 1)
			# -----------------------
			
			nearest_indices = distances.argsort()[:k]
			nearest_distances = distances[nearest_indices]
			
			# List "nearest_label" have form [3, 4, 8, 3, 1] (when k = 5), 
			# and function should return 3 in that case.
			nearest_labels = self.ytr[nearest_indices]
			
			# You have to implement strategy in case of nearest_label have multiple majority.
			# (ex. top 5 nearest neighbors are [3, 3, 4, 9, 4])
			# select predictions from prediction list 

			# implement kNN using nearest_indices and nearest_distances below.
			# ----------------------
			# it only returns first nearest element.
			Ypred[i] = nearest_labels[0]
			# ----------------------

		return Ypred
