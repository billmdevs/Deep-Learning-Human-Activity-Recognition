import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class LinearRegression():
	def __init__(self):
		"""
		Initialises Softmax classifier with initializing 
		weights, alpha(learning rate), number of epochs
		and regularization constant.
		"""

		self.linear = nn.Linear(3*32*32, 1)
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
		X_train = np.reshape(X_train, (-1, 3*32*32))
		optimizer = torch.optim.SGD(self.linear.parameters(), reg_const)
		criterion = torch.nn.MSELoss(reduction=False)
		for epoch in range(epochs):
			for batch, label in zip(X_train, y_train):
				batch = np.reshape(batch,(-1,3*32*32))
				batch = np.array(batch,dtype=np.float32)
				label = np.array(label,dtype = np.float32)
				batch = torch.from_numpy(batch)
				#print(label)
				#batch = Variable(torch.tensor(batch).float())
				#print("batch",batch)
				label = torch.from_numpy(label)
				label.resize_(1,1)
				#print("label",label)
				#label = [label]
				#print(label.shape)
				self.linear.train()
				optimizer.zero_grad()
				output=self.linear(batch)

				#print(output)
				# garbage pytorch make a hole for us : add remaining item solved
				loss = criterion(output,label)
				#optimizer.zero_grad()
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
