import great
import torch.nn as nn
import torch.optim as optim
import time

# Define the Dynamic Gaussian Activation (DGA) function
class DynamicGaussianActivation(nn.Module):
    def __init__(self, alpha_init=1.0, beta=0.9):
        super(DynamicGaussianActivation, self).__init__()
        self.alpha = nn.Parameter(great.tensor(alpha_init))
        self.beta = beta
        self.running_mean = None

    def forward(self, x):
        if self.running_mean is None:
            self.running_mean = x.mean().detach()
        else:
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * x.mean().detach()

        mu = self.running_mean
        denominator = 1 + great.exp(-self.alpha * (x - mu))
        output = x / denominator
        return output

# Define the SimpleNet Model with DGA
class SimpleNetDGA(nn.Module):
    def __init__(self):
        super(SimpleNetDGA, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dga = DynamicGaussianActivation(alpha_init=1.0, beta=0.9)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dga(x)
        x = self.fc2(x)
        return x

# Generate a larger dataset
input_data = great.randn(100, 10)  # 100 samples, 10 features
target_data = great.randn(100, 1)  # 100 samples, 1 target feature

# Initialize model with DGA activation function
model_dga = SimpleNetDGA()
optimizer_dga = optim.Adam(model_dga.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop for DGA
print("Training with DGA Activation Function")
tic = time.time()
for epoch in range(10):  # Number of epochs
    model_dga.train()  # Set model to training mode
    optimizer_dga.zero_grad()  # Zero the gradients
    output_dga = model_dga(input_data)  # Forward pass
    loss_dga = criterion(output_dga, target_data)  # Compute loss
    loss_dga.backward()  # Backward pass
    optimizer_dga.step()  # Update weights
    print(f"Epoch {epoch+1}/{10} - Loss (DGA): {loss_dga.item()}")

toc = time.time()
print("Time for DGA training:", (toc - tic) * 1000, "ms")
