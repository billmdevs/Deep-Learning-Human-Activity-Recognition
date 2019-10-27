import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import data_process
from torch.autograd import Variable
class LinearRegression(torch.nn.Module):

    input_size = 3*32*32
    num_class = 10
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.5
        self.epochs = 5
        self.reg_const = 0.05
        self.linear = torch.nn.Linear(3*32*32,1)
    def forward(self,image):
        imgae = image.reshape(-1,3*32*32)
        out = self.linear(image)
        return out
    def accuracy(self,l1,l2):
        return torch.sum(l1==l2).item()/len(l1)
    def train(self, X_train, y_train):
        """
        Train Linear regression classifier using function from Pytorch
        """
        X_train = np.reshape(X_train,(-1,3*32*32))
        #print(self.linear.weight.shape)
        iter = 0

        optimizer = torch.optim.SGD(self.linear.parameters(), lr=0.01)
        loss_fn = torch.nn.L1Loss(size_average=True)
        for i in range(0,10):
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
                loss = loss_fn(output,label)
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter = iter + 1
                if iter%10000 == 0:

                #print(loss.item())
                    print(loss.item(),label)
    
    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        pred = 1
        return pred


if __name__ == '__main__':

    data = data_process.get_CIFAR10_data()
    X_train = data[ 'X_train']/(510)
    y_train = data['y_train']
    print("data_ready")

    model = LinearRegression()
    model.train(X_train,y_train)
    test = data["X_test"][0]/510
    test = np.reshape(test,(-1,3*32*32))
    test = np.array(test, dtype=np.float32)
    test = torch.from_numpy(test)
    print("predict",model.linear(test),"label",data["y_test"][0])
