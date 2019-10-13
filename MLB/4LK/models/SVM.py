import numpy as np

class SVM():
	def __init__(self):
		"""
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
		"""
		self.w = None
		self.alpha = 0.01
		self.epochs = 100
		self.reg_const = 0.05
        
	def calc_gradient(self, X_train, y_train):
		"""
		It is not mandatory for you to implement this function if you find an equivalent one in Pytorch
		
		
          Calculate gradient of the svm hinge loss.
          
          Inputs have dimension D, there are C classes, and we operate on minibatches
          of N examples.

          Inputs:
          - X_train: A numpy array of shape (N, D) containing a minibatch of data.
          - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
            that X[i] has label c, where 0 <= c < C.

          Returns:
          - gradient with respect to weights W; an array of same shape as W
		"""
		W = self.w
		grad_w = np.zeros(W.shape)
		num_classes = W.shape[1]
		num_train = X_train.shape[0]

		for i in range(num_train):
			scores = X_train[i].dot(W)
			correct_class_score = scores[y_train[i]]
			summed_indicator_functions = 0
			for j in range(num_classes):
				if j == y_train[i]:
					continue
				margin = scores[j] - correct_class_score + 1
				if margin > 0:
					grad_w[:, j] += X_train[i]
					summed_indicator_functions += 1
			grad_w[:, y_train[i]] -= summed_indicators_functions*X_train[i]
		grad_w = num_train
		
		reg = self.reg_const
		grad_w += 2 * reg * W

		return grad_w
        
	def train(self, X_train, y_train):
		"""
        Train SVM classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
		"""
		num_train, dim = X_train.shape
		num_classes = np.max(y_train) + 1
		if self.w is None:
			self.w = 0.001 * np.random.randn(dim, num_classes)
		
		loss = []
		for it in range(num_iters):
			batch_idx = np.random.choice(num_train, batch_size)
			X_batch = X[batch_idx]
			y_batch = y[batch_idx]

	def predict(self, X_test):
		"""
        Use the trained weights of svm classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
		"""
		
		pred = np.zeros(X_test.shape[0])
		pred = np.argmax(np.dot(X, self.w), axis=1)

		return pred
