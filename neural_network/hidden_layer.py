import numpy as np

class HiddenLayer(Layer):
    def __init__(self, num_neuron, previous_layer, activation_function = "sigmoid"):
        """
        Bias and weight are initialised randomly using mean = 0, variance = 1
        """
        super().__init__(num_neuron, activation_function = activation_function)
        self.previous_layer: Layer = previous_layer
        self.weights = np.array([np.random.randn(num_neuron, previous_layer.output.shape[0])])
        self.input = np.zeros((previous_layer.num_neuron, 1))
        self._init_activation_function(activation_function)