import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('/home/ben-mulder/Documents/Python/VIX.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')  # Sort by date
df.set_index('Date', inplace=True)  # Set Date as index

# Rename columns if necessary
df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Create features
df['Returns'] = df['Close'].pct_change()
df['High_Low_Ratio'] = df['High'] / df['Low']
df['Close_Open_Ratio'] = df['Close'] / df['Open']

# Drop NaN values
df.dropna(inplace=True)

# Select features for the model
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'High_Low_Ratio', 'Close_Open_Ratio']
X = df[features]
Y = df['Close'].shift(-1)  # Predict next day's closing price

# Remove last row (no target value for last day)
X = X[:-1]
Y = Y[:-1]

# Convert to NumPy arrays
X = X.values
Y = Y.values

print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

print(f"first Y: {Y[0]}")
print(f"second Y: {Y[1]}")
print(f"third Y: {Y[2]}")


# Get the index for the split point
split_index = int(0.8 * len(X))

# Split the data sequentially
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

print(f"Shape of X train: {X_train.shape}")
print(f"Shape of Y train: {Y_train.shape}")

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessed and ready for the neural network.")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

#mini batches



def init_params(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters


def forward_prop(X, parameters, activation="relu"):
    cache = {"A0": X}
    L = len(parameters) // 2
    
    for l in range(1, L):
        Z = parameters[f"W{l}"] @ cache[f"A{l-1}"] + parameters[f"b{l}"]
        cache[f"Z{l}"] = Z
        
        if activation == "relu":
            A = np.maximum(Z, 0)
        elif activation == "gelu":
            A = 0.5 * Z * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * Z**3)))
        else:
            raise ValueError("Unsupported activation function")
        
        cache[f"A{l}"] = A
    
    ZL = parameters[f"W{L}"] @ cache[f"A{L-1}"] + parameters[f"b{L}"]
    AL = ZL  # Linear activation for output layer
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = AL
    
    return AL, cache



def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

def gelu_derivative(Z):
    sqrt_2_pi = np.sqrt(2 / np.pi)
    tanh_part = np.tanh(sqrt_2_pi * (Z + 0.044715 * np.power(Z, 3)))
    term1 = 0.5 * (1 + tanh_part)
    sech2_part = 1 - np.power(tanh_part, 2)
    term2 = 0.5 * Z * sech2_part * (sqrt_2_pi * (1 + 0.134145 * np.power(Z, 2)))
    return term1 + term2


def back_prop(Y, AL, cache, parameters, activation):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[0]  # Number of samples
    dAL = AL - Y
    grads[f"dZ{L}"] = dAL

    for l in reversed(range(1, L + 1)):
        grads[f"dW{l}"] = 1/m * grads[f"dZ{l}"] @ cache[f"A{l-1}"].T
        grads[f"db{l}"] = 1/m * np.sum(grads[f"dZ{l}"], axis=1, keepdims=True)
        
        if l > 1:
            dA_prev = parameters[f"W{l}"].T @ grads[f"dZ{l}"]
            if activation == "relu":
                grads[f"dZ{l-1}"] = dA_prev * relu_derivative(cache[f"Z{l-1}"])
            elif activation == "gelu":
                grads[f"dZ{l-1}"] = dA_prev * gelu_derivative(cache[f"Z{l-1}"])
            else:
                raise ValueError("Unsupported activation function")
    
    return grads


def update_parameters(parameters, grads, learning_rate=0.001):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters


def nn(X, Y, layer_dims, activation, num_iterations=1000, learning_rate=0.001, batch_size=64):
    parameters = init_params(layer_dims)
    costs = []
    m = X.shape[0]
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    for i in range(num_iterations):
        epoch_cost = 0  # Reset cost for this epoch
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]
            
            AL, cache = forward_prop(X_batch.T, parameters, activation)
            grads = back_prop(Y_batch, AL, cache, parameters, activation)
            parameters = update_parameters(parameters, grads, learning_rate)
            
            # Accumulate cost for each mini-batch
            batch_cost = np.mean(np.square(AL - Y_batch))
            epoch_cost += batch_cost

        # After completing all mini-batches, average the epoch cost
        epoch_cost /= (m // batch_size)
        costs.append(epoch_cost)

        # Optionally print the cost every 10 epochs
        if i % 10 == 0:
            print(f"Cost after epoch {i}: {epoch_cost}")

    
    return AL, cache, parameters, costs




# Set up the neural network
input_dim = X_train_scaled.shape[1]
layer_dims = [input_dim, 64, 32, 16, 1]

#Hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.0001
ACTIVATION = "gelu"
BATCH_SIZE = 64


# Train the model
Y_pred, cache, parameters,costs = nn(X_train_scaled, Y_train, layer_dims, ACTIVATION, EPOCHS, LEARNING_RATE,BATCH_SIZE)

# Evaluate predictions on the test set
Y_pred_test, _ = forward_prop(X_test_scaled.T, parameters, ACTIVATION)
mse = np.mean(np.square(Y_pred_test - Y_test.reshape(1, -1)))
print(f"Test Mean Squared Error: {mse}")


# Calculate R-squared
Y_pred_test = Y_pred_test.flatten()

ss_res = np.sum(np.square(Y_test - Y_pred_test))
ss_tot = np.sum(np.square(Y_test - np.mean(Y_test)))
r_squared = 1 - (ss_res / ss_tot)
print(f"Test R-squared: {r_squared}")

# Plot learning curve
import matplotlib.pyplot as plt
plt.plot(costs)
plt.xlabel('Iterations (x100)')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.show()
