import numpy as np
import math
import h5py
import json

from forwardLayer import ForwardLayer
from activationFunctions import tanh, tanh_prime
from losses import mse, mse_prime 
from activationLayer import ActivationLayer

class Network:
    
    #constructor for the class
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def addLayer(self, layer):
        self.layers.append(layer)

    # set loss to use
    def useLoss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        self.state = result
        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    #saves the current predicted value in a JSON-file
    def save(self, fileName):

        #creating an emptpy list to store the objects with the weights for every forwardlayer
        data = []
        for layer in self.layers:
            if(isinstance(layer, ForwardLayer)):
                jsonLine = {'weights': layer.getWeights().tolist()}
                data.append(jsonLine)

        #save the weights in a json file
        with open (fileName, 'w') as file:
            json.dump(data, file)

    #loading a modelstate from a json file
    def load(self, fileName):

        with open (fileName, 'r') as jsonFile:
            file = json.load(jsonFile)
            #replacing for every forwardlayer the save weights from the json file
            #find all forwardLayers 
            
            i = 0

            for layer in self.layers:

                if(isinstance(layer, ForwardLayer)):
                    layer.weights = np.array(file[i]['weights'])
                    i += 1
      
