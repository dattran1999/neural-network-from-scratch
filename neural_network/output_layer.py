import numpy as np
from hidden_layer import HiddenLayer

class OutputLayer(HiddenLayer):
    def __init__(self, num_neuron, activation_function = "sigmoid"):
        super().__init__(num_neuron, activation_function = activation_function)

    def output_error(self, actual_output):
        self.error = np.multiply(self.error_func_prime(actual_output), self.act_func_prime(self.zs))
    
    def error_func_prime(self, actual_output):
        return np.subtract(actual_output, self.output)
