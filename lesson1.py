import numpy as np


def calc(inputs, weights, biases):
    list_output = []
    for n_weights, n_bias in zip(weights, biases):
        n_output = 0
        for n_input, weight in zip(inputs, n_weights):
            n_output += weight * n_input
        n_output += n_bias
        list_output.append(n_output)
    return list_output


def dot_prod(inputs, weights, biases):
    return np.dot(inputs, np.array(weights).T) + biases


inputs = [
    [2, 5, 1, 7],
    [4, 1, 6, 9],
    [1, 4, 6, 3],
]

weights = [
    [1.0, 0.3, 0.7, -0.2],
    [0.5, 0.8, -0.2, 0.1],
    [0.4, -0.1, 0.3, 1.0],
]

bias = [3, 4, 6, 1]

# output = (
#     inputs[0] * weights[0]
#     + inputs[1] * weights[1]
#     + inputs[2] * weights[2]
#     + inputs[3] * weights[3]
#     + bias
# )

output = dot_prod(inputs, weights, bias)

print(output)