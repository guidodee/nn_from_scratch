# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
import numpy as np
from tqdm import tqdm

'''TO DO
    1. Add abiltiy to use different activation functions
    2. Train on actual data
    '''

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
   return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y, length_layers, no_of_layers):
        self.input = x
        self.no_of_layers = no_of_layers + 1
        self.weights = [0] * self.no_of_layers
        self.layers = [0] * self.no_of_layers
        self.layers[0] = self.input
        self.length_layers = length_layers

        for i in range(1, self.no_of_layers):
            self.weights[i-1] = np.random.rand(self.layers[i-1].shape[1], self.length_layers)
            self.layers[i] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
        self.weights[-1] = np.random.randn(self.length_layers, y.shape[1])

        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        for i in range(1, self.no_of_layers):
            self.layers[i]= sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
        self.output = sigmoid(np.dot(self.layers[i], self.weights[i]))

    def backpropropgate(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        for i in range(self.no_of_layers-1, -1, -1):
            if i == self.no_of_layers-1:
                # print(i)
                propogate = 2*(self.y - self.output) * sigmoid_derivative(self.output)
                # print(propogate.shape)
                # print(self.layers[i].T.shape)
                # print(self.weights[i].shape)
                self.weights[i] += np.dot(self.layers[i].T, propogate)  
            else:
                # print(i)
                # print(propogate.shape)
                # print(self.weights[i+1].T.shape)
                # print(self.weights[i].shape)
                # np.dot(propogate, self.weights[i+1].T)
                # np.dot(propogate, self.weights[i+1].T)* sigmoid_derivative(self.layers[i+1])
                self.weights[i] += np.dot(self.layers[i].T, (np.dot(propogate, self.weights[i+1].T)* sigmoid_derivative(self.layers[i+1])))
                propogate = (np.dot(propogate, self.weights[i+1].T)* sigmoid_derivative(self.layers[i+1]))
        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # # update the weights with the derivative (slope) of the loss function
        # self.weights1 += d_weights1
        # self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y, 4, 1)

    for i in tqdm(range(100000)):
        nn.feedforward()
        nn.backpropropgate()

    print(nn.output)

    # print(nn.output)

    # print(sigmoid_derivative(nn.output))