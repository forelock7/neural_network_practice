import numpy as np

np.random.seed(0)

X = [[1, 4, 5, 6], [1, 3, 2, 1], [5, 8, 3, 2]]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 + np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = LayerDense(4, 8)
layer2 = LayerDense(8, 8)
layer3 = LayerDense(8, 2)

layer1.forward(X)
layer2.forward(layer1.output)
layer3.forward(layer2.output)

print(layer3.weights)