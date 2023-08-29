import numpy as np
from mlp import Multilayer
from keras.datasets import mnist

"""################## setup data ##################"""

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = np.reshape(train_x, (60000, 784, 1))
train_x = train_x.astype(np.float64)        

test_x = np.reshape(test_x, (10000, 784, 1))
test_x = test_x.astype(np.float64)

(train_x, test_x) = (train_x/255, test_x/255)

def num_hot_encode(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

train_y = [num_hot_encode(y) for y in train_y]
test_y = [num_hot_encode(y) for y in test_y]

"""#################### run MLP ####################"""

mlp = Multilayer([784, 120, 50, 120, 784])
p, decay_rate, batch_size, epochs = 0.25, 0.97, 20, 200
mlp.run_SGD(epochs, train_x, train_y, test_x, test_y, batch_size, p, decay_rate)


