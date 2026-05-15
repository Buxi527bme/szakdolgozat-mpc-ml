import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

# --- 1. A Dinamikus Modellhez igazított Hálózat ---
class TinyMPCNet(nn.Module):
    def __init__(self):
        super(TinyMPCNet, self).__init__()
        # 4 bemenet -> 32 neuron -> 32 neuron -> 1 kimenet
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def train():
    print("Dinamikus adatok betöltése...")
    df = pd.read_csv('data/mpc_dynamic_dataset.csv')
    
    # Bemenetek: e_y, e_vy, e_psi, e_r | Kimenet: delta
    X = df[['e_y', 'e_vy', 'e_psi', 'e_r']].values
    y = df[['delta']].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
        
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Tanítás ezen: {device}")
    
    model = TinyMPCNet().to(device)
    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Kicsit finomabb tanulási ráta
    epochs = 300 # Több epoch a bonyolultabb adatokhoz
    
    print("Tanítás indítása...")
    for epoch in range(epochs):
        model.train()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_tensor)
                test_loss = criterion(test_preds, y_test_tensor)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")
            
    torch.save(model.state_dict(), 'models/tinympc_ai_weights.pth')
    print("Tanítás kész! A modell már a dinamikus összefüggéseket használja.")

if __name__ == "__main__":
    train()