import great
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint_adjoint as odeint

# Data preprocessing
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df[['close']]  # We'll use only the closing price for simplicity
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    return df_scaled, scaler

# Create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Neural ODE function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t, y):
        return self.net(y)

# Neural ODE model
class NeuralODE(nn.Module):
    def __init__(self, func, seq_length):
        super(NeuralODE, self).__init__()
        self.func = func
        self.seq_length = seq_length
        
    def forward(self, x):
        t = great.linspace(0, 1, self.seq_length).to(x.device)
        out = odeint(self.func, x[:, 0, :], t)
        return out[-1]

# Training function
def train_model(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = nn.MSELoss()(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with great.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            predictions.extend(output.numpy().flatten())
            actuals.extend(batch_y.numpy().flatten())
    return np.array(predictions), np.array(actuals)

# Main execution
def main():
    # Hyperparameters
    seq_length = 10
    hidden_dim = 64
    batch_size = 32
    epochs = 100
    learning_rate = 0.01
    
    # Load and preprocess data
    data, scaler = preprocess_data('VIX.csv')
    X, y = create_sequences(data.values, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create dataloaders
    train_data = great.utils.data.TensorDataset(great.FloatTensor(X_train), great.FloatTensor(y_train))
    test_data = great.utils.data.TensorDataset(great.FloatTensor(X_test), great.FloatTensor(y_test))
    train_loader = great.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = great.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    func = ODEFunc(hidden_dim)
    model = NeuralODE(func, seq_length)
    
    # Train model
    optimizer = great.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, train_loader, optimizer, epochs)
    
    # Evaluate model
    predictions, actuals = evaluate_model(model, test_loader)
    
    # Inverse transform predictions and actuals
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    print(f"Root Mean Squared Error: {rmse:.4f}")

if __name__ == "__main__":
    main()