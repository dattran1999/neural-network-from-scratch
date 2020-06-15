import numpy as np

class Layer():
    def __init__(self, num_neuron):
        """
        Bias and weight are initialised randomly using mean = 0, variance = 1
        """
        self.num_neuron: int = num_neuron
        self.bias = np.zeros((num_neuron,))
        self.input = np.array([])
        self.output = np.zeros((num_neuron,))

    def feed_forward(self):
        raise NotImplementedError("Must implement in HiddenLayer or InputLayer")

    def set_next_layer(self, layer):
        self.next_layer = layer
    
    def log(self):
        print("shape of input", self.input.shape)
        print(f"Input of layer: ", self.input)
        print(f"shape of output", self.output.shape)
        print(f"Output of layer: ", self.output)
