import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0        # Total time
N = 1000       # Number of steps
dt = T / N     # Time step
t = np.linspace(0, T, N+1)  # Time grid

# Simulate Brownian increments
dW = np.random.normal(0, np.sqrt(dt), size=N)

# Initialize Brownian path
W = np.zeros(N+1)
W[1:] = np.cumsum(dW)

# Plot the path
plt.figure(figsize=(10,6))
plt.plot(t, W, label='Brownian Motion Path')
plt.title('Simulation of Brownian Motion')
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.legend()
plt.grid(True)
plt.show()

