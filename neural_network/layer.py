import numpy as np

def Layer():
    def __init__(self, num_neuron, activation_function = "sigmoid"):
        """
        Bias and weight are initialised randomly using mean = 0, variance = 1
        """
        self.num_neuron: int = num_neuron
        self.bias = np.array([np.random.randn(num_neuron, 1)])
        self.input = np.array([])
        self.output = np.zeros((num_neuron, 1))

    def feed_forward(self):
        raise NotImplementedError("Must implement in HiddenLayer or InputLayer")
