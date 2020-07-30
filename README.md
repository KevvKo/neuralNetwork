# Neural Network
Just in development.
## Table of Contents
[Requirements](#requirements)

[Description](#description)

[Getting Started](#getting-started)

[Coming soon](#coming-soon)
## Requirements
### Technically
    - Python3++

### Modules
- numpy


## Description
This is a simple neural network, using currently a feedword layer with a backpropagation. Furthermore, the sigmoid function is as activion implemented. 

The neuralNetwork is basically not developed to solve complex problems, it is just a simple example, how to build a neural network from scratch.

## Getting Started
Create a new instance of Network:

```python
network = Network()
```

After creating an instance from the network, add a desired number of layers:
```python
net.addLayer(ForwardLayer(1, 4))
net.addLayer(ActivationLayer(tanh, tanh_prime))
net.addLayer(ForwardLayer(4, 8))
net.addLayer(ActivationLayer(tanh, tanh_prime))
net.addLayer(ForwardLayer(8, 1))
net.addLayer(ActivationLayer(tanh, tanh_prime))
```

A full example to build a model, you can find in the **neuralNetwork.py-file**

## Coming soon
This neural network will be extended with time. It should be capable to solve complex problems, using multiple algorithms for deep learning. Also the current available functions can be handle different shapes of arrays. 
