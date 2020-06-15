from neural_network import NeuralNetwork
from input_layer import InputLayer
from output_layer import OutputLayer
from hidden_layer import HiddenLayer

# test train xor function 
model = NeuralNetwork(
             InputLayer(2),
             HiddenLayer(2, "sigmoid"),
             OutputLayer(1, "sigmoid")
        )
train_input = [[1, 1], [1, 0], [0, 1], [0, 0]]
train_output = [0, 1, 1, 0]
model.train(train_input, train_output, 1, 0.1, 20)
model.predict([1, 1])
model.predict([1, 0])
model.predict([0, 1])
model.predict([0, 0])
