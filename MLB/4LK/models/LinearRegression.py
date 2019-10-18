import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class LinearRegression(torch.nn.Module):
	def __init__(self):
		"""
		Initialises Softmax classifier with initializing 
		weights, alpha(learning rate), number of epochs
		and regularization constant.
		"""

		super(LinearRegression, self).__init__()
		self.linear = torch.nn.Linear(1, 1)
		self.w = None
		self.alpha = 0.5
		self.epochs = 100
		self.reg_const = 0.05
	
    
	def train(self, X_train, y_train):
		"""
		Train Linear regression classifier using function from Pytorch
		"""
		epochs = self.epochs
		reg_const = self.reg_const
		criterion = torch.nn.MSELoss(size_average=False)
		optimizer = torch.optim.SGD(super(LinearRegression, self).parameters(), reg_const)

		for epoch in range(epochs):
			super(LinearRegression, self).train()
			optimizer.zero_grad()

			# Forward pass
			pred = linreg(X_train)

			# Compute Loss
			loss = criterion(pred, y_train)

			# Backward pass
			loss.backward()
			optimizer.step()


    
	def predict(self, X_test):
		"""
		Use the trained weights of softmax classifier to predict labels 		for data points.

		Inputs:
		- X_test: A numpy array of shape (N, D) containing training data;
		there are N training samples each of dimension D.

		Returns:
		- pred: Predicted labels for the data in X_test. pred is a 1-	dimensional array of length N, and each element is an integer giving the predicted class.
        """
		pred = self.linear(X_test)


		return pred
