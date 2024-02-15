import numpy as np
import math
# inputs = [1, 2, 3, 4]
# weights = [[ 0.74864643, -1.00722027,  1.45983017,  1.34236011],
#            [-1.20116017, -0.08884298, -0.46555646,  0.02341039],
#            [-0.30973958,  0.89235565, -0.92841053,  0.12266543]]
# biases = [0, 0.3, -0.5]
# layer_outputs = [] # we create the list that will contain the results of the processing of the neurons of the layer
# # for each neuron
# for neuron_weights, neuron_bias in zip(weights, biases):
#     # initialize output to 0
#     neuron_output = 0
#     # for each input and weight
#     for n_input, weight in zip(inputs, neuron_weights):
#         # multiply input and weight and add it to the output
#         neuron_output += n_input * weight
#         print("neuron_output: ", neuron_output)
#     # add bias to the output
#     neuron_output += neuron_bias
#     print("neuron_output_bias: ", neuron_output)
#     # add the neuron result to the layer
#     layer_outputs.append(neuron_output)
# print(layer_outputs) # print the result

def linear(x):
    return x

def sigmoid(z):
    return 1/(1+ math.exp(-z))

def relu(x): # Rectified Linear Unit
    return max(0, x)

def leaky_relu(a, x): # Leaky Rectified Linear Unit
    return max(a*x, x)

def elu(a, x): # Exponential Linear Unit
    return a*(math.exp(x) - 1) if x < 0 else x

def softplus(x):
    return math.log(1 + math.exp(x))

input = [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [7, 8, 9, 10]]

output = []
for vector in input:
    output_vector = []
    for element in vector:
        print("element: ", sigmoid(element))
        output_vector.append(sigmoid(element))
    output.append(output_vector)
print("output: ", np.reshape(output, [3,4]))


