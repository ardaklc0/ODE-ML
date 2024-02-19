import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['ggplot'])


X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1) # y = 8 + 24x + noise to make it more realistic and challenging
# y = 4 + 3 * X # y = 4 + 3x (x is element of X = 2 * np.random.rand(100,1))

# plt.plot(X,y,'b.')
# plt.ylabel('Y')
# plt.xlabel('X')
# plt.show()

learning_rate = 0.001
iterations = 10000

# We initiate the wanted theta values with random values to fit theta value
theta = np.random.randn(2,1)
print("Theta: ", end='\n')
print(theta, end='\n')

X_b = np.c_[np.ones((len(X),1)),X]
print("X_b: ", end='\n')
print(X_b, end='\n')
print("------------------------------------------\n")
# theta,cost_history,theta_history = gradient_descent(X_b, y, theta,lr,n_iter)

m = len(y) # no of samples
cost_history = np.zeros(iterations) # np.zeros creates an array of zeros with the given shape
theta_history = np.zeros((iterations,2)) # np.zeros creates an array of zeros with the given shape
for i in range(iterations):
    prediction = np.dot(X_b, theta)
    error = prediction - y
    theta = theta - (1/m) * learning_rate * (X_b.T.dot((error)))
    theta_history[i,:] = theta.T
    cost_history[i] = (1/2*m) * np.sum(np.square(prediction-y))


print("Cost History: ", end='\n')
print(cost_history, end='\n')
print("Theta History: ", end='\n')
print(theta_history, end='\n')
print("------------------------------------------\n")
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
