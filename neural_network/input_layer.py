import numpy as np

class InputLayer(Layer):
    def __init__(self, num_neuron):
        super().__init__(num_neuron)
        self.input = np.zeros((num_neuron, 1))

    def feed_forward(self):
        self.output = self.input
