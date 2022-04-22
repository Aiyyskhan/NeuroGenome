import numpy as np


class NeuralNet:
    def __init__(self, weight_list):
        self.W0 = weight_list[0]
        self.W1 = weight_list[1]
        self.W2 = weight_list[2]

        self.relu = lambda x: (np.absolute(x) + x) / 2

    def __call__(self, input_data):
        input_data = input_data[:,None,:]	
        potentials = self.relu(np.tanh(np.matmul(input_data, self.W0)))
        potentials = self.relu(np.tanh(np.matmul(potentials, self.W1)))
        output = self.relu(np.tanh(np.matmul(potentials, self.W2)))
        return output.reshape((output.shape[0], output.shape[2]))