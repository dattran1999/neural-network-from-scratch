import random
import numpy as np
from input_layer import InputLayer
from output_layer import OutputLayer
from layer import Layer
from typing import List

class NeuralNetwork:
    def __init__(self, *args):
        """
        Create a neural network with specied Layer objects in args
        First layer (first argument) has to be input layer
        Last layer (last argument) has to be output layer
        """
        self.layers = []
        if not isinstance(args[0], InputLayer):
            raise TypeError("First layer must be of type InputLayer")
        if not isinstance(args[-1], OutputLayer):
            raise TypeError("Last layer must be of type OutputLayer")
        for arg in args:
            if not isinstance(arg, Layer):
                raise TypeError("Arguments must be of type Layer")
            self.layers.append(arg)

    def init(self):
        self.init_layers()

    def init_layers(self):
        # set previous layer and next layer for each layer
        for i, layer in enumerate(self.layers):
            if i != 0:
                layer.set_previous_layer(self.layers[i-1])
            if i != len(self.layers) - 1:
                layer.set_next_layer(self.layers[i+1])
        
    def train(self, training_input, training_output, mini_batch_size, eta, epochs = 5):
        self.init()
        if len(training_input) != len(training_output):
            raise IndexError("Training input must have same size as training output")

        def randomised_training_batch(x, y):
            training_data = [(x[i], y[i]) for i in range(len(x))]
            random.shuffle(training_data)
            n = len(training_data)
            for i in range(0, n, mini_batch_size):
                yield training_data[i : i+mini_batch_size]

        for i in range(epochs):
            print(f"Epoch {i + 1}")
            for mini_batch in randomised_training_batch(training_input, training_output):
                self._train(mini_batch, eta)
    
    def _train(self, mini_batch, eta):
        # sum of δC/δw of corresponding layer 
        sum_dC_dw = []
        # sum of δC/δb of corresponding layer 
        sum_dC_db = []
        for i, layer in enumerate(self.layers):
            sum_dC_db.append(np.zeros((layer.num_neuron,)))
            if i != 0:
                sum_dC_dw.append(np.zeros((layer.num_neuron, layer.previous_layer.output.shape[0])))
            else:
                sum_dC_dw.append(np.zeros((layer.num_neuron,)))
        for x, y in mini_batch:
            print(x, y)
            # feed forward in every layer
            for i, layer in enumerate(self.layers):
                if i == 0:
                    layer.read_input(x)
                layer.feed_forward()
            
            # error of output
            self.layers[-1].output_error(y)
            print(self.layers[-1].error)

            # backpropagation from last layer to first hidden layer
            for i in range(len(self.layers)-2, 1, -1):
                delta_b, delta_w = self.layers[i].back_propagation()
                sum_dC_dw[i] += delta_w
                sum_dC_db[i] += delta_b

            # update weights and biases 
            for i in range(len(self.layers)-1, 1, -1):
                self.layers[i].update_weights_and_biases(sum_dC_db[i], sum_dC_dw[i], len(mini_batch), eta)

    def predict(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.read_input(x)
            layer.feed_forward()
        print(self.layers[-1].output)
        return layer.output
