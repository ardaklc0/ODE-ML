import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['ggplot'])

def plot_GD(n_iter,lr,ax,ax1=None):
     """
     n_iter = no of iterations
     lr = Learning Rate
     ax = Axis to plot the Gradient Descent
     ax1 = Axis to plot cost_history vs Iterations plot

     """
     _ = ax.plot(X,y,'b.')
     theta = np.random.randn(2,1)

     tr =0.1
     cost_history = np.zeros(n_iter)
     for i in range(n_iter):
        pred_prev = X_b.dot(theta)
        theta,h,_ = gradient_descent(X_b,y,theta,lr,1)
        pred = X_b.dot(theta)

        cost_history[i] = h[0]

        if ((i % 25 == 0) ):
            _ = ax.plot(X,pred,'r-',alpha=tr)
            if tr < 0.8:
                tr = tr+0.2
     if not ax1== None:
        _ = ax1.plot(range(n_iter),cost_history,'b.')  

def calculate_cost(theta, X, y):
    '''
    Calculate the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))
    
    where:
        j is the no of features
    '''
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    '''
    X     = Matrix of X with added bias units
    y     = Vector of Y
    theta = Vector of thetas np.random.randn(j,1)
    learning_rate = Amount of learning rate
    iterations = No of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y) # no of samples
    cost_history = np.zeros(iterations) # np.zeros creates an array of zeros with the given shape
    theta_history = np.zeros((iterations,2)) # np.zeros creates an array of zeros with the given shape
    for i in range(iterations):
        prediction = np.dot(X, theta) 
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[i,:] = theta.T
        cost_history[i]  = calculate_cost(theta,X,y)

    print(theta)
    print(cost_history)
    print(theta_history)

    return theta, cost_history, theta_history

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

# Graph setup
# fig = plt.figure(figsize=(10,8),dpi=100)
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# it_lr =[(2000,0.001),(500,0.01),(200,0.05),(100,0.1)]
# count =0
# for n_iter, lr in it_lr:
#     count += 1
    
#     ax = fig.add_subplot(4, 2, count)
#     count += 1
   
#     ax1 = fig.add_subplot(4,2,count)
    
#     ax.set_title("lr:{}".format(lr))
#     ax1.set_title("Iterations:{}".format(n_iter))
#     plot_GD(n_iter,lr,ax,ax1)
# plt.show()