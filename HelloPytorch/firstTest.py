import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # convolutions:
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # here using two distinct layers to test
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # 6 input image channel, 16 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Pooling layers
        self.pool1  = nn.MaxPool2d(kernel_size=2)
        self.pool2  = nn.MaxPool2d(kernel_size=2)
        # activations
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        # flatten whole tensor
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        # 1*32*32 -> 6*32*32
        x1 = self.conv11(x)
        x2 = self.conv12(x)
        x = torch.cat((x1,x2),dim=1)
        #x = self.conv1(x)
        # activation
        x = self.relu1(x)
        # 6*32*32 -> 6*16*16
        x = self.pool1(x)
        # 6*16*16 -> 16*14*14
        x = self.conv2(x)
        # activation
        x = self.relu2(x)
        # 16*14*14 -> 16*7*7 (last column/row is dropped)
        x = self.pool2(x)
        # 16*7*7 -> 1*576
        x = self.flatten1(x)
        # 1*576 -> 1*120
        x = self.fc1(x)
        # activation
        x = self.relu3(x)
        # 1*120 -> 1*84
        x = self.fc2(x)
        # activation
        x = self.relu4(x)
        # 1*84 -> 10
        x = self.fc3(x)
        return x

seed = 42
torch.manual_seed(seed)

net = Net()

input = torch.randn(1, 1, 32, 32)
output = net(input)
print(output)

optimizer = optim.SGD(params=net.parameters(), lr=0.1)
criterion =   nn.MSELoss()

target = torch.randn(10)  # a dummy target, for example
target = torch.zeros([1,10],dtype=torch.float32)
target[0,0] = 1
target = target.view(1, -1)  # make it the same shape as output


for epoch in range(0,1):
  optimizer.zero_grad()
  output=net(input)
  target = target.view(1, -1)  # make it the same shape as output
  loss=criterion(output,target)
  print(loss.grad_fn)
  print(loss)
  loss.backward()
  optimizer.step()

print(net(input))