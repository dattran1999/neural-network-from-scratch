from exceptions.exceptions import InputError
import numpy as np
from layer import Layer

class InputLayer(Layer):
    def __init__(self, num_neuron):
        super().__init__(num_neuron)
        self.input = []

    def read_input(self, x):
        x = np.array(x)
        if x.shape != (self.num_neuron,):
            raise InputError("Shape of actual input is different from number of neuron in InputLayer")
        self.input = x

    def feed_forward(self):
        self.output = self.input
