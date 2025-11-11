import torch as q
import numpy as np
import torchvision.datasets as datasets
import random
import matplotlib.pyplot as plt

# КОНВОЛЮЦИОННАЯ СЕТЬ

# выставляем одинаковые сиды, чтобы не было изменяющихся радомных данных
random.seed(0)
q.manual_seed(0)
q.cuda.manual_seed(0)
np.random.seed(0)
# для восроизводимости результатов CUDA Deep Neural Network library при работе с гпу
q.backends.cudnn.deterministic = True

mnist_train = datasets.MNIST(root='./data', train=True)
mnist_test = datasets.MNIST(root='./data', train=False)

# TRAIN
x_train = mnist_train.data
y_train = mnist_train.targets
# TEST
x_test = mnist_test.data
y_test = mnist_test.targets

print("studying:\n", "pictures: ", len(x_train), "; numbers: ", len(y_train), sep='')

plt.imshow(x_train[0])
plt.title(f"number: {y_train[0]}")
plt.show()
print("number: ", y_train[0])

x_train = x_train.unsqueeze(1)  # Tensor: 60000*28*28 -> 60000*1*28*28
y_train = y_train.unsqueeze(1)
x_train = x_train.float()
x_test = x_test.float()


class LeNet(q.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # изображение 2d

        # свертка: размер ядра - 3*3, отступ - 0, шаг 1, вых.каналы - 6 -> вых.каналы - 6
        # ВМЕСТО СВЕРТКИ 5*5 БЕРЕМ 2 СВЕРТКИ 3*3 ДЛЯ ИСПОЛЬЗОВАНИЯ МЕНЬШИХ ВЕСОВ 25w vs 18w
        self.conv1_1 = q.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.conv1_2 = q.nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=0)
        # F.act. - ReLU
        self.act1 = q.nn.ReLU()
        # нормируем кучу
        self.norm1 = q.nn.BatchNorm2d(num_features=6)
        # пулинг 2*2, шаг 2
        self.pool1 = q.nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling

        self.conv2_1 = q.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = q.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        # self.conv2 = q.nn.Conv2d(6, 16, 5)  # stride 1, padding 0
        self.act2 = q.nn.ReLU()
        self.norm2 = q.nn.BatchNorm2d(num_features=16)
        self.pool2 = q.nn.MaxPool2d(kernel_size=2, stride=2)

        # Для линейного вектора:
        self.fc_linear1 = q.nn.Linear(in_features=16 * 5 * 5, out_features=120)  # 16*5*5 - длина вектора после свертки
        self.act3 = q.nn.ReLU()

        self.fc_linear2 = q.nn.Linear(in_features=120, out_features=84)
        self.act4 = q.nn.ReLU()

        self.fc_linear3 = q.nn.Linear(in_features=84, out_features=10)

    def forward(self, batch):
        batch = self.conv1_2(self.conv1_1(batch))
        batch = self.act1(batch)
        batch = self.norm1(batch)
        batch = self.pool1(batch)

        batch = self.conv2_2(self.conv2_1(batch))
        batch = self.act2(batch)
        batch = self.norm2(batch)
        batch = self.pool2(batch)

        # преобразование многомерного тензора в двумерный для подачи f.c.linear
        # -1 -- автоматическое вычисление длины вектора по второй оси
        batch = batch.view(batch.size(0), -1) 

        batch = self.fc_linear1(batch)
        batch = self.act3(batch)
        batch = self.fc_linear2(batch)
        batch = self.act4(batch)
        batch = self.fc_linear3(batch)

        return batch


net = LeNet()
loss_f = q.nn.CrossEntropyLoss()
optimizer = q.optim.Adam(net.parameters(), lr=0.001)

device = q.device("cuda:0" if q.cuda.is_available() else "cpu")
net.to(device)
x_train = x_train.to(device)
x_test = x_test.to(device)

accuracy_history_test = []
loss_history = []

batch_size = 100
for epoch in range(10):
    order = np.random.permutation(len(x_train))

    # обучение батчами
    for idx in range(0, len(order), batch_size):
        optimizer.zero_grad()
        # подбираются параметры мат. ожидания и старндартного отклонения ф. в слое
        # ставим флаг на обучение
        # параметры нормализации подстраиваются не при градиентном спуске (убывании ф. потерь), а при прямом вычеслении в forward
        net.train()

        batch_indexes = order[idx:idx + batch_size]

        x_batch = x_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        prediction = net.forward(x_batch)

        loss_value = loss_f(prediction, y_batch)
        loss_value.backward()

        optimizer.step()
    # чтобы при тестировании сети не изменяись параметры нормализации мы ставим флаг evaluation
    net.eval()

    test_prediction = net.forward(x_test)

    # .data -- получаем скаляр графа
    loss_history.append(loss_f(prediction, y_test).data.cpu())

    accuracy = (test_prediction.argmax(dim=1) == y_test).float().mean().data.cpu()
    accuracy_history_test.append(accuracy)

    if epoch % 2 == 0:
        print(accuracy)

# Graphic
plt.plot(accuracy_history_test)
plt.show()
plt.plot(loss_history)
plt.show()

