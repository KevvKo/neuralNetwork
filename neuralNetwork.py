
import numpy as np

from network import Network
from forwardLayer import ForwardLayer
from activationFunctions import tanh, tanh_prime
from losses import mse, mse_prime 
from activationLayer import ActivationLayer

# training data
x_train = np.array([[0], [0], [1], [1]])
y_train = np.array([[1], [1], [1], [0]])

# network
net = Network()
net.addLayer(ForwardLayer(1, 256))
net.addLayer(ActivationLayer(tanh, tanh_prime))
net.addLayer(ForwardLayer(256, 128))
net.addLayer(ActivationLayer(tanh, tanh_prime))
net.addLayer(ForwardLayer(128, 1))
net.addLayer(ActivationLayer(tanh, tanh_prime))

#
# net.load('modelState.json')

# train
net.useLoss(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)

net.save('modelState.json')
