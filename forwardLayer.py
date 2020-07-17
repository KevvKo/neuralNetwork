import numpy as np
from layer import Layer

#inherit from the abstract class layer
class ForwardLayer(Layer):
    #input_size = number of input neurons
    #output_size = number of output neurons

    def __init__(self, inputSize, outputSize):
        self.weights = np.random.rand(inputSize, outputSize) - 0.5
        self.bias = np.random.rand(1, outputSize) -0.5

    #returns the output for a given input
    def forward_propagation(self, inputData):
        self.input = inputData
        self.output = np.dot(self.input, self.weights) + self.bias 
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights
