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
        self._init_activation_function(activation_function)

    def _init_activation_function(self, function_name):
        """Set the activation function according to the name given"""
        if function_name == "sigmoid":
            self.activation_function = self.sigmoid
        else:
            # TODO: Raise error
            pass
        
    def sigmoid(self, x):
        """Sigmoid function"""
        return 1/(1 + np.exp(-x))

    def update_input(self):
        """Set input of current layer to be previous layer's output"""
        self.input = self.previous_layer.output

    def feed_forward(self):
        """
            Obtain previous layer output and set as input.
            Then compute output using output = weight * input + bias
        """
        self.update_input()
        self.output = self.activation_function(np.add(np.dot(self.weights, self.input) , self.bias))

    def back_propogation(self):
        pass

    
