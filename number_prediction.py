import torch as t
import random
import numpy as np

random.seed(0)
np.random.seed(0)
t.manual_seed(0)
t.cuda.manual_seed(0)
t.backends.cudnn.determenistic = True

import torchvision.datasets

mnist_train = torchvision.datasets.MNIST('./', download=True, train=True)
mnist_test = torchvision.datasets.MNIST('./', download=True, train=False)

x_train = mnist_train.data
y_train = mnist_train.targets
x_test = mnist_test.data
y_test = mnist_test.targets

x_test = x_test.float()
x_train = x_train.float()

print('Изначальная размерность:', x_train.shape, x_test.shape)

# меняем размерность для удобства обучения кучами
x_train = x_train.reshape([-1, 28 * 28])
x_test = x_test.reshape([-1, 28 * 28])
print('новая размерность:', x_train.shape, x_test.shape)


# НЕЙРОННАЯ СЕТЬ
class MNISTnet(t.nn.Module):
    def __init__(self, hidden_neurons):
        super(MNISTnet, self).__init__()
        self.layer1 = t.nn.Linear(28 * 28, hidden_neurons)
        self.act1 = t.nn.Sigmoid()
        self.layer2 = t.nn.Linear(hidden_neurons, 10)  # классификация на 10 чисел

    def forward(self, study_tensor):
        study_tensor = self.layer1(study_tensor)
        study_tensor = self.act1(study_tensor)
        study_tensor = self.layer2(study_tensor)
        # для ускорения обучения не используем софтмакс
        return study_tensor


number_net = MNISTnet(100)

print('60k картинок, 784 пикселя в каждой: ', x_train.shape)
print('60k цифр [0,9] для каждой картинки: ', y_train.shape)

# ВЫБОР НОСИТЕЛЯ ДЛЯ ОБРАБОТКИ (ЦПУ / ГПУ)
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
number_net = number_net.to(device)
x_test = x_test.to(device)
x_train = x_train.to(device)
y_test = y_test.to(device)
y_train = y_train.to(device)

# ф. кросс-энтропия принимает на вход выходной н.слой из forward()
f_loss = t.nn.CrossEntropyLoss()  # - Ф. потерь для многоклассовой классификации
optiMethod = t.optim.Adam(number_net.parameters(),
                          lr=0.001)  # передаются все параметры н.с. (оптимизируются веса н.слоев)

loss_rates = []
score_of_success_rates = []

batch = 100
for epoch in range(1, 201):
    perm_i = np.random.permutation(len(x_train))

    for ind in range(0, len(x_train), batch):
        optiMethod.zero_grad()

        batch_indexes = perm_i[ind:ind + batch]
        x_batch = x_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        prediction = number_net.forward(x_batch)

        # стохастический градиентный спуск
        loss = f_loss(prediction, y_batch)
        loss.backward()
        optiMethod.step()

    test_prediction = number_net.forward(x_test)
    loss_rates.append(f_loss(test_prediction, y_test).to('cpu').detach().numpy())

    # среднее количество угадываний
    score_of_success = (test_prediction.argmax(
        dim=1) == y_test).float().mean()  # argmax(dim=1) проверяет у какого нейрона выход ближе всего к 1(вероятности)
    score_of_success_rates.append(score_of_success.to('cpu').detach().numpy())

    if epoch % 20 == 0:
        print('EPOCH: ', epoch, ';  QUALITY: ', score_of_success, sep='')



import matplotlib.pyplot as plt

plt.plot(loss_rates, c='red')
plt.xlabel('epoch')
plt.ylabel('коэф. потерь таргетных значений')

plt.plot(score_of_success_rates)
plt.ylabel('% верных предсказаний')
plt.xlabel('epoch')

plt.show()
