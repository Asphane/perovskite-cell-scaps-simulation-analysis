import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# Define the MLP Architecture
class JV_Model(nn.Module):
    def __init__(self, input_dim):
        super(JV_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    pkl_path = 'Machine_Learning_Dashboard/jv_curve_dataset.pkl'
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    print("Loading PyTorch J-V dataset...")
    df = pd.read_pickle(pkl_path)
    
    # Define features (same as before)
    targets_meta = ['PCE', 'Voc', 'Jsc', 'FF', 'Source_File', 'V_list', 'J_list']
    phys_features = [col for col in df.columns if col not in targets_meta]
    print(f"Physical Features ({len(phys_features)}):", phys_features)
    
    # Handle NaNs using median imputation
    imputer = SimpleImputer(strategy='median')
    df[phys_features] = imputer.fit_transform(df[phys_features])
    
    # We need to flatten the V_list and J_list to create continuous data rows
    # Each row becomes multiple rows: one for each (V, J) point.
    print("Flattening J-V curves...")
    flat_data = []
    
    # We will only extract points from the illuminated IV curves (where J <= 0, but since SCAPS output might be positive/negative, we'll just take all points for now)
    for idx, row in df.iterrows():
        v_list = row['V_list']
        j_list = row['J_list']
        
        # Base features for this simulation
        base_feats = row[phys_features].values
        
        for v, j in zip(v_list, j_list):
            # Input: [Phys_Params..., V] -> Output: [J]
            flat_row = np.append(base_feats, v)
            flat_data.append((flat_row, j))
            
    flat_data = np.array(flat_data, dtype=object)
    
    X = np.vstack(flat_data[:, 0]).astype(np.float32)
    y = np.vstack(flat_data[:, 1]).astype(np.float32)
    
    print(f"Total training data points: {X.shape[0]}")
    
    # Standardize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled)
    y_tensor = torch.tensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Initialize Model
    model = JV_Model(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training PyTorch MLP (this may take a few seconds)...")
    epochs = 20
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {epoch_loss:.4f}")
        
    # Save the PyTorch Model & Scaler & Imputer
    os.makedirs('Machine_Learning_Dashboard/pytorch_assets', exist_ok=True)
    
    torch.save(model.state_dict(), 'Machine_Learning_Dashboard/pytorch_assets/jv_mlp.pth')
    joblib.dump(scaler, 'Machine_Learning_Dashboard/pytorch_assets/scaler.pkl')
    joblib.dump(imputer, 'Machine_Learning_Dashboard/pytorch_assets/imputer.pkl')
    joblib.dump(phys_features, 'Machine_Learning_Dashboard/pytorch_assets/phys_features.pkl')
    
    print("\nSuccessfully trained and saved PyTorch MLP model!")

if __name__ == "__main__":
    main()
