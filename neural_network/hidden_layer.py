import numpy as np
from layer import Layer

class HiddenLayer(Layer):
    def __init__(self, num_neuron, activation_function = "sigmoid"):
        """
        TODO: write documentation for attributes and methods
        Bias and weight are initialised randomly using mean = 0, variance = 1
        """
        super().__init__(num_neuron)
        # z = w * a + b, basically the result before applying our activation function to get the output
        self.zs = np.zeros((num_neuron,1))
        self._init_activation_function(activation_function)

    def _init_activation_function(self, function_name):
        """Set the activation function according to the name given"""
        if function_name == "sigmoid":
            self.act_func = self.sigmoid
            self.act_func_prime = self.sigmoid_prime
        else:
            # TODO: Raise error
            pass

    def set_previous_layer(self, previous_layer):
        self.previous_layer: Layer = previous_layer
        self.weights = np.random.randn(self.num_neuron, previous_layer.output.shape[0])
        self.input = np.zeros((previous_layer.num_neuron, 1))

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        """Derivative of sigmoid function"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def update_input(self):
        """Set input of current layer to be previous layer's output"""
        self.input = self.previous_layer.output

    def feed_forward(self):
        """
            Obtain previous layer output and set as input.
            Then compute output using output = weight * input + bias
        """
        self.update_input()
        # ("print-------------------------")
        # print("before feedforward")
        # self.log()
        self.zs = np.dot(self.weights, self.input) + self.bias
        self.output = self.act_func(self.zs)
        # print("-------------------------")
        # print("after feedforward")
        # self.log()

    def back_propogation(self):
        w = self.next_layer.weights
        next_layer_error = self.next_layer.error
        self.error = np.multiply(np.dot(np.transpose(w), next_layer_error) , self.sigmoid_prime(self.zs))
        delta_w = np.dot(self.error, self.previous_layer.output.transpose())
        return (self.error, delta_w)

    def update_weights_and_biases(self, sum_dC_db, sum_dC_dw, batch_size, eta):
        # TODO:
        self.weights -= eta/batch_size * sum_dC_dw
        self.bias -= eta/batch_size * sum_dC_db

    def log(self):
        print("shape of input", self.input.shape)
        print(f"Input of layer: ", self.input)
        print(f"shape of output and z", self.output.shape, self.zs.shape)
        print("shape of bias", self.bias.shape)
        print(f"Output of layer: ", self.output)