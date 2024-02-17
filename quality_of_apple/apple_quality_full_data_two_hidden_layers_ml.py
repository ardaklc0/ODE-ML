import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm 
import statistics
from sklearn.metrics import confusion_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def show_normal_dist_wrt_column_name(column_name, data, title):
    x_values_train = range(len(data))
    mean_train = statistics.mean(x_values_train)
    sd_train = statistics.stdev(x_values_train)
    # Plotting the normalized training data
    plt.plot(x_values_train, norm.pdf(x_values_train, mean_train, sd_train), color='red', label=title)
    plt.xlabel('Data Points')
    plt.ylabel(column_name)
    plt.title(column_name)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

# Load the data
csv_file_path = 'quality_of_apple/apple_quality.csv'
df = pd.read_csv(csv_file_path)

# Omit id column
data = df.drop(['A_id', 'Quality'], axis=1)
expected_output = df['Quality']

# Taking 0.33 of the data for training
unnormalized_train_data = data
# Normalize the training data
train_data = (unnormalized_train_data - unnormalized_train_data.mean()) / unnormalized_train_data.std()
# show_normal_dist_wrt_column_name('Size', unnormalized_train_data['Size'], 'Unnormalized Train Size')


inputs = np.array(train_data) # Taking inputs
epochs = 10000 # Number of iterations
lr = 0.01 # Learning rate
expected_output_as_integer = (df['Quality'] == 'good').astype(int) # Convert the expected output to integers
expected_output_as_integer_train = expected_output_as_integer
expected_output_as_integer_train = expected_output_as_integer_train.to_numpy(int).reshape(4000,1)

np.random.seed(5)

# Define the number of neurons in each hidden layer
hidden_neurons_layer1 = 20
hidden_neurons_layer2 = 7

# Initialize weights and biases for the first hidden layer
hidden_weights_layer1 = np.random.uniform(size=(inputs.shape[1], hidden_neurons_layer1))
hidden_bias_layer1 = np.random.uniform(size=(1, hidden_neurons_layer1))

# Initialize weights and biases for the second hidden layer
hidden_weights_layer2 = np.random.uniform(size=(hidden_neurons_layer1, hidden_neurons_layer2))
hidden_bias_layer2 = np.random.uniform(size=(1, hidden_neurons_layer2))

output_weights = np.random.uniform(size=(hidden_neurons_layer2, 1))
output_bias = np.random.uniform(size=(1, 1))

# Graph setup
# fig, ax = plt.subplots()
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.set_title('Loss Function over Epochs')

losses = []

for epoch in range(epochs):
     # Forward Propagation
    hidden_layer1_activation = np.dot(inputs, hidden_weights_layer1) + hidden_bias_layer1
    hidden_layer1_output = sigmoid(hidden_layer1_activation)

    hidden_layer2_activation = np.dot(hidden_layer1_output, hidden_weights_layer2) + hidden_bias_layer2
    hidden_layer2_output = sigmoid(hidden_layer2_activation)

    output_layer_activation = np.dot(hidden_layer2_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Calculate loss
    error = expected_output_as_integer_train - predicted_output
    loss = np.mean(0.5 * (error ** 2))
    losses.append(loss)

    # Gradient Descent
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Backpropagation for the second hidden layer
    error_hidden_layer2 = d_predicted_output.dot(output_weights.T)
    d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer2_output)
    
    # Backpropagation for the first hidden layer
    error_hidden_layer1 = d_hidden_layer2.dot(hidden_weights_layer2.T)
    d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer1_output)

    # Update Weights and Biases
    output_weights = output_weights + lr * hidden_layer2_output.T.dot(d_predicted_output)
    output_bias = output_bias + lr * np.sum(d_predicted_output)
    
    hidden_weights_layer2 = hidden_weights_layer2 + lr * hidden_layer1_output.T.dot(d_hidden_layer2)
    hidden_bias_layer2 = hidden_bias_layer2 + lr * np.sum(d_hidden_layer2)
    
    hidden_weights_layer1 = hidden_weights_layer1 + lr * inputs.T.dot(d_hidden_layer1)
    hidden_bias_layer1 = hidden_bias_layer1 + lr * np.sum(d_hidden_layer1)

    if epoch % (epochs // 10) == 0:
        progress = epoch / epochs * 100
        print(f"Training Progress: [{int(progress)}%] {'>' * int(progress / 10)}{'.' * (10 - int(progress / 10))}", end='\r')



# Plot the loss function
# lt.plot(range(epochs), losses)
# plt.show()

# Prediction
print("\nOutput from neural network after 10,000 epochs: ", end='\n')
predicted_output_modified = np.array([1 if value > 0.5 else 0 for value in predicted_output]).reshape(4000,1)
print(predicted_output_modified)

# Comparison
comparison = pd.DataFrame({'Expected': expected_output_as_integer_train.flatten(), 'Predicted': predicted_output_modified.flatten(), }, )
comparison['Match'] = comparison['Expected'] == comparison['Predicted']
print(comparison)

# Confusion Matrix
conf_matrix = confusion_matrix(expected_output_as_integer_train, predicted_output_modified)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Assuming conf_matrix is your confusion matrix
TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TN = conf_matrix[1, 1]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Precision
precision = TP / (TP + FP)
print(f'Precision: {precision * 100:.2f}%')
# Recall (Sensitivity)
recall = TP / (TP + FN)
print(f'Recall (Sensitivity): {recall * 100:.2f}%')
# Specificity
specificity = TN / (TN + FP)
print(f'Specificity: {specificity * 100:.2f}%')

# If you want to plot the confusion matrix, you can use matplotlib
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()