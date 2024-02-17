import numpy as np
import matplotlib.pyplot as plt

"""
The oldest problem in the neural network is the XOR problem. The XOR problem is a two-class classification problem.
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def forward_propagation(inputs, weights, bias):
    return np.dot(inputs, weights) + bias
def backward_propagation(error, output):
    return error * sigmoid_derivative(output)
np.random.seed(5)
epochs = 10000 # Number of iterations
lr = 0.1 # Learning rate
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
expected_output = np.array([[0],
                            [1],
                            [1],
                            [1]])
hidden_weights = np.random.uniform(size=(2,2))
hidden_bias =np.random.uniform(size=(1,2))
output_weights = np.random.uniform(size=(2,1))
output_bias = np.random.uniform(size=(1,1))

# Set up interactive mode
# plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Function over Epochs')

losses = []

input_text = ax.text(0.2, 0.95, '', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color='black', fontsize=8)
prediction_text = ax.text(0.6, 0.95, '', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color='red', fontsize=8)
expected_text = ax.text(0.9, 0.95, '', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', color='blue', fontsize=8)

for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Calculate loss
    error = expected_output - predicted_output
    loss = np.mean(0.5 * (error ** 2))
    losses.append(loss)

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights = output_weights + lr * hidden_layer_output.T.dot(d_predicted_output)
    output_bias = output_bias + lr * np.sum(d_predicted_output)
    hidden_weights = hidden_weights + lr * inputs.T.dot(d_hidden_layer)
    hidden_bias = hidden_bias + lr * np.sum(d_hidden_layer)

    # Update the plot
    # ax.plot(range(epoch + 1), losses, color='blue')
    # prediction_text.set_text(f'Prediction: {predicted_output.flatten()}')
    # expected_text.set_text(f'Expected: {expected_output.flatten()}')
    # fig.canvas.draw()
    # fig.canvas.flush_events()

    # Progress Bar
    # progress_percentage = (epoch + 1) / epochs * 100
    # progress_bar = "[" + "=" * int(progress_percentage // 5) + ">" + " " * (20 - int(progress_percentage // 5)) + "]"
    # print(f"\rEpoch {epoch + 1}/{epochs} {progress_bar} {progress_percentage:.2f}%", end="")

# Turn off interactive mode at the end
# plt.ioff()
input_text.set_text(f'Inputs: {inputs}')
prediction_text.set_text(f'Prediction: {predicted_output}')
expected_text.set_text(f'Expected: {expected_output}')
plt.plot(range(epochs), losses)
plt.show()

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*[1 if value > 0.5 else 0 for value in predicted_output])