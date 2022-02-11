import numpy as np

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
   return x * (1.0 - x)

def relu(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = max(0.0, x[i][j])  
    return x

def relu_derivative(x): 
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] >= 0:
                x[i][j] = 1
            else:
                x[i][j] = 0
    return x

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - (np.tanh(x))**2
