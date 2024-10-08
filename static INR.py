"static INR trained to generalize volatility surface over all the datapoints"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # Import tqdm for progress bar

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess the volatility surface data
def load_volatility_surface_data(file_path):
    """
    Load the implied volatility surface data from a CSV file.
    The file should contain columns: Date, Tenor, 0.1, 0.2, ..., 1.9.
    """
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv('training_data.csv')

    # Convert tenor (e.g., '2M', '1Y') to numeric years
    def tenor_to_years(tenor):
        if 'M' in tenor:
            return float(tenor.replace('M', '')) / 12
        elif 'Y' in tenor:
            return float(tenor.replace('Y', ''))
        return float(tenor)
    
    df['Tenor'] = df['Tenor'].apply(tenor_to_years)

    # Moneyness levels (as they appear in the CSV, e.g., 0.1, 0.2, ..., 1.9)
    moneyness_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

    # Prepare the inputs (strike, maturity) and outputs (implied volatility)
    inputs = []
    outputs = []
    
    for _, row in df.iterrows():
        tenor = row['Tenor']
        for i, moneyness in enumerate(moneyness_levels):
            inputs.append([moneyness, tenor])
            outputs.append(row[i + 2])  # Implied volatility data starts from column 2 (0.1 moneyness)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return torch.from_numpy(inputs).float(), torch.from_numpy(outputs).float()

# Volatility surface model
class VolatilitySurfaceNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=5):
        super(VolatilitySurfaceNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))  # Output implied volatility
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)  # Output shape: (batch_size,)

def train_vol_surface_network(model, optimizer, criterion, inputs, vol, epochs=100, batch_size=64):
    inputs = inputs.to(device)
    vol = vol.to(device)
    dataset = torch.utils.data.TensorDataset(inputs, vol)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0  # Keep track of the number of processed batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_inputs, batch_vol in progress_bar:
            batch_inputs = batch_inputs.to(device)
            batch_vol = batch_vol.to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_vol)
            loss.backward()
            optimizer.step()

            # Accumulate the total loss over the epoch
            epoch_loss += loss.item() * batch_inputs.size(0)
            batch_count += 1

            # Update the progress bar description with the accumulated average loss so far
            avg_loss_so_far = epoch_loss / (batch_count * batch_size)
            progress_bar.set_postfix({"Avg Loss": avg_loss_so_far})
        
        # Calculate the final average loss for the epoch
        epoch_loss /= len(dataloader.dataset)

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")


# Visualization function
def visualize_vol_surface(model, grid_size=50):
    moneyness_levels = np.linspace(0.1, 1.9, grid_size)
    maturities = np.linspace(0.1, 40, grid_size)
    grid_moneyness, grid_maturity = np.meshgrid(moneyness_levels, maturities)
    grid_points = np.stack([grid_moneyness, grid_maturity], axis=-1).reshape(-1, 2)
    grid_tensor = torch.from_numpy(grid_points).float().to(device)

    with torch.no_grad():
        vol_pred = model(grid_tensor).cpu().numpy()

    vol_grid = vol_pred.reshape(grid_size, grid_size)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_moneyness, grid_maturity, vol_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Tenor (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Learned Implied Volatility Surface')
    plt.show()

# Main function
def main():
    file_path = 'volatility_surface.csv'  # Replace with the path to your CSV file
    epochs = 10
    batch_size = 64
    learning_rate = 1e-4

    # Load real volatility surface data
    inputs, vol = load_volatility_surface_data(file_path)

    # Initialize model, optimizer, and loss function
    model = VolatilitySurfaceNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train the model
    train_vol_surface_network(model, optimizer, criterion, inputs, vol, epochs=epochs, batch_size=batch_size)

    # Visualize the learned volatility surface
    visualize_vol_surface(model, grid_size=50)

if __name__ == "__main__":
    main()
