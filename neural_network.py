# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
import numpy as np
from tqdm import tqdm
from activation_functions import *
import matplotlib.pyplot as plt


class NeuralNetwork:
    '''A Python implementation of a Neural Network
    '''
    def __init__(self, x, y, length_layers, no_of_layers, activate_function=sigmoid, a_f_derivative= sigmoid_derivative):
        self.input = x
        self.no_of_layers = no_of_layers + 1
        self.weights = [0] * self.no_of_layers
        self.biases = [0] * self.no_of_layers
        self.layers = [0] * self.no_of_layers
        self.layers[0] = self.input
        self.length_layers = length_layers
        self.activate_function = activate_function
        self.a_f_derivative = a_f_derivative
        self.y = y
        self.loss_tracker = []

        for i in range(1, self.no_of_layers):
            self.weights[i-1] = np.random.rand(self.layers[i-1].shape[1], self.length_layers)
            self.layers[i] = self.activate_function(np.dot(self.layers[i-1], self.weights[i-1]))
            self.biases[i-1] = np.random.rand(*self.layers[i].shape)
        self.weights[-1] = np.random.randn(self.length_layers, self.y.shape[1])



    def feedforward(self):
        for i in range(1, self.no_of_layers):
            self.layers[i]= self.activate_function(np.dot(self.layers[i-1], self.weights[i-1])+self.biases[i-1])
        self.output = self.activate_function(np.dot(self.layers[i], self.weights[i])+self.biases[i])

    def backpropropgate(self):
        for i in range(self.no_of_layers-1, -1, -1):
            if i == self.no_of_layers-1:
                weight_propogate = 2*(self.y-self.output) * self.a_f_derivative(self.output)
                self.weights[i] += np.dot(self.layers[i].T, weight_propogate)  
            else:
                self.weights[i] += np.dot(self.layers[i].T, (np.dot(weight_propogate, self.weights[i+1].T)* self.a_f_derivative(self.layers[i+1])))
                weight_propogate = (np.dot(weight_propogate, self.weights[i+1].T)* self.a_f_derivative(self.layers[i+1]))
        self.track_loss(sum((self.y - self.output)**2))
    
    def train(self, iterations):
        for i in tqdm(range(iterations)):
            self.feedforward()
            self.backpropropgate()
        print(f'The predicted values are {self.output}')
        plt.scatter(list(range(0, iterations)), self.loss_tracker)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.show()

    def track_loss(self, loss):
        self.loss_tracker.append(loss)
