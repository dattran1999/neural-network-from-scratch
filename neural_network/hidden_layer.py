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
        # z = w * a + b, basically the result before applying our activation function to get the output
        self.zs = np.zeroes((num_neuron, 1))
        self._init_activation_function(act_func)

    def _init_activation_function(self, function_name):
        """Set the activation function according to the name given"""
        if function_name == "sigmoid":
            self.act_func = self.sigmoid
            self.act_func_prime = self.sigmoid_prime
        else:
            # TODO: Raise error
            pass

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        """Derivative of sigmoid function"""
        return sigmoid(x) * (1 - sigmoid(x))

    def update_input(self):
        """Set input of current layer to be previous layer's output"""
        self.input = self.previous_layer.output

    def feed_forward(self):
        """
            Obtain previous layer output and set as input.
            Then compute output using output = weight * input + bias
        """
        self.update_input()
        self.zs = np.add(np.dot(self.weights, self.input) , self.bias)
        self.output = self.act_func(self.zs)

    def set_next_layer(self, layer):
        self.next_layer = layer

    def back_propogation(self):
        w = self.next_layer.weights
        next_layer_error = self.next_layer.error
        self.error = np.multiply(np.dot(np.transpose(w), next_layer_error) , sigmoid_prime(self.zs))
