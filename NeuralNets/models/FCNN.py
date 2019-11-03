import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# Hyperparameters
epochs = 10
num_classes = 10
batch_size = 200
learning_rate = 0.01
log_interval = 10

def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)

# create a two hidden layers fully connected network
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, 10)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x)

net = Net()
print(net)

# create a stochastic gradient descent optimizer
optimizer = optim.Adamax(net.parameters(), lr=learning_rate)
# create a loss function
criterion = nn.NLLLoss()

# run the main training loop
for epoch in range(epochs):
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data, target
		# resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
		#data = data.view(data.shape[0], -1)		
		data = data.view(-1, 32*32*3)
		optimizer.zero_grad()
		net_out = net(data)
		loss = criterion(net_out, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data))

# run a test loop
test_loss = 0
correct = 0
for data, target in test_loader:
	data, target = Variable(data, volatile=True), Variable(target)
	data = data.view(-1, 32*32*3)
	net_out = net(data)
	# sum up batch loss
	test_loss += criterion(net_out, target).data
	pred = net_out.data.max(1)[1]  # get the index of the max log-probability
	correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
