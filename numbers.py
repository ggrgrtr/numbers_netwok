import torch as q
import numpy as np
import torchvision.datasets as datasets
import random

random.seed(0)
q.manual_seed(0)
q.cuda.manual_seed(0)
np.random.seed(0)
q.backends.cudnn.determenistic = True

mnist_train = datasets.MNIST(root='./data',train=True)
mnist_test = datasets.MNIST(root='./data',train=False)

# TRAIN
x_train=mnist_train.data
y_train=mnist_train.targets
# TEST
x_test=mnist_test.data
y_test=mnist_test.targets


x_train.unsqueeze(1)
y_train.unsqueeze(1)
x_train=x_train.float()
x_test=x_test.float()


class LeNet(q.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5) #?


    def forward(self, x):




        return x



lenet5=LeNet()

loss_f=q.nn.CrossEntropyLoss()
optimizer=q.optim.Adam(model.parameters(),lr=0.001)



device = q.device("cuda:0" if torch.cuda.is_available() else "cpu")
lenet5.to(device)
x_train=x_train.to(device)
x_test=x_test.to(device)




accuracy_history_test=[]
loss_history=[]

batch_size=100
for epoch in range(10000):
    order=np.random.permutation(len(x_train))
    for idx in range(0,len(order),batch_size):
        optimizer.zero_grad()
        batch_indexes=order[idx:idx+batch_size]

        x_batch=x_train[batch_indexes].to(device)
        y_batch=y_train[batch_indexes].to(device)

        prediction=lenet5.forward(x_batch)

        loss_value=loss_f(prediction,y_batch)
        loss_value.backward()

        optimizer.step()

    test_prediction=lenet5.forward(x_test)
    loss_history.append(loss_f(prediction,y_test).data.cpu())

    accuracy=(test_prediction.argmax(dim=1)==y_test).float().mean().data.cpu()
    accuracy_history_test.append(accuracy)

    if epoch % 100 == 0:
        print(accuracy)

import matplotlib.pyplot as plt
plt.plot(accuracy_history_test)
plt.show()
plt.plot(loss_history)
plt.show()