нормализация, дропаут, последовательная свёрка
используя две последовательные свертки 3х3, вместо одной 5х5 мы уменьшаем количество передаваемых весов в слое, что улучшает % верных предсказаний, но и ускоряет переобучение

# Нейронные сети на основе LeNet

## Сверточная нейронная сеть с нормализацией для CIFAR

**LeNet**:

*MaxPooling, AvgPooling, ReLU, BatchNormalization, Convolution(5x5), 2x[Convolution(3x3)]*

<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/36ca4a6b-b0af-49fc-83bb-17d3215e7cc1" />
<img width="556" height="435" alt="image" src="https://github.com/user-attachments/assets/d8adbe45-3412-41ca-8f73-b406b443fac5" />

На основе первого графика можно увидеть, что:

**MaxPooling > AvgPooling** -- для изображений

Batch-нормализация сильно ускоряет процесс обучения (и переобучения)

Две свертки 3х3 дают бОльшую погрешность, чем свертка 5х5. Из-за меньшего кол-ва весов у 3х3, сеть стала хуже обучаться (и быстрее переобучаться)

**! Не для всех задач подходят стандартные оптимальные решения, рекомендации носят эмпирический характер**


## Свёрточная нейронная сеть для MNIST

-> *numbers_conv.py*


## Полносвязанная нейронная сеть для определения чисел без свертки MNIST

-> *number_prediction_fully_connected.py*

или 0_9_fc.ipynb

<img width="1001" height="680" alt="image" src="https://github.com/user-attachments/assets/7fbd9917-53a4-468e-95f1-c25cc4636d28" />


Изменение функции потерь и точности предсказания в зависимости от итерируемой эпохи

<img width="421" height="325" alt="image" src="https://github.com/user-attachments/assets/5bb85eab-1079-4995-a27c-ca559b0ae679" />



<img width="617" height="433" alt="image" src="https://github.com/user-attachments/assets/dc537177-7027-4c6e-965c-0f0826620073" />
