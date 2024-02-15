import matplotlib.pyplot as plt
import numpy as np
import math

#dy/dt = y' = y = f(t, y)

#t_0
t_0 = 0

#y_0
y_0 = 1
y_values = []


#step_size := h
step_size = 0.1
epoch = 100

for i in range(epoch):
    y_0 = y_0 + step_size * y_0
    y_values.append(y_0)

print("y_values:", y_values)


# # Create a list of time values, assuming t_0 = 0
# time_values = [i * step_size for i in range(epoch)]

# # Plot the calculated 'y' values versus time
# plt.plot(time_values, y_values)

# # Labels and title
# plt.xlabel("Time (t)")
# plt.ylabel("y")
# plt.title("Exponential Growth Simulation")

# ex_values = [math.exp(x) for x in time_values]

# # Plotting both results
# plt.plot(time_values, y_values, label="Simulation")
# plt.plot(time_values, ex_values, label="e^x")


# # Show the plot
# plt.show()