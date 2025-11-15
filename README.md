# Нейронные сети на основе LeNet

## Сверточная нейронная сеть с нормализацией для CIFAR

-> *CIFAR_test.ipynb*

**LeNet и MNISTnet**:

*MaxPooling, AvgPooling, ReLU, BatchNormalization, Convolution(5x5), 2x[Convolution(3x3)]*

<img width="552" height="435" alt="image" src="https://github.com/user-attachments/assets/ac87c082-8608-4027-8e6b-d485bec31b80" />
<img width="561" height="435" alt="image" src="https://github.com/user-attachments/assets/09ff5a57-72b6-429b-bb62-0c341f13829b" />

LeNet хорошо справляется с предссказанием одноканальных тензоров (цифры), но не с RGB.

CIFAR_net имеет больше каналов (до 256), что улучшает качество из-за извлечения из данных большего количества информации.

На основе первого графика можно увидеть, что:

**MaxPooling > AvgPooling** -- для изображений

Batch-нормализация сильно ускоряет процесс обучения (и переобучения)

Bспользуя две последовательные свертки 3х3, вместо одной 5х5 мы уменьшаем количество передаваемых весов в слое, что улучшает % верных предсказаний, но и ускоряет переобучение
Две свертки 3х3 дают бОльшую погрешность, чем свертка 5х5. В данном случае из-за меньшего кол-ва весов у 3х3, сеть стала хуже обучаться (и быстрее переобучаться)

Обычно, когда у модели, ннаоборот, слишком много параметров, она становится склонной к переобучению

**! Не для всех задач подходят стандартные оптимальные решения, рекомендации носят эмпирический характер**


## Свёрточная нейронная сеть для MNIST

-> *numbers_conv.py*

<img width="807" height="690" alt="image" src="https://github.com/user-attachments/assets/e9349433-3ada-4343-89ee-8a2fd70a0bb2" />
<img width="806" height="691" alt="image" src="https://github.com/user-attachments/assets/16dfcbf1-48df-45d2-994d-d718d9e15fa2" />

Из-за использования батч-нормализации сеть быстро стала очень уверенна в своих ответах, в следствии чего сильно возросла функция потерь



## Полносвязанная нейронная сеть для определения чисел без свертки MNIST

-> *number_prediction_fully_connected.py*

или 0_9_fc.ipynb

<img width="1001" height="680" alt="image" src="https://github.com/user-attachments/assets/7fbd9917-53a4-468e-95f1-c25cc4636d28" />


Изменение функции потерь и точности предсказания в зависимости от итерируемой эпохи

<img width="421" height="325" alt="image" src="https://github.com/user-attachments/assets/5bb85eab-1079-4995-a27c-ca559b0ae679" />



<img width="617" height="433" alt="image" src="https://github.com/user-attachments/assets/dc537177-7027-4c6e-965c-0f0826620073" />
