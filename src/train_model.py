import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

class TinyMPCNet(nn.Module):
    def __init__(self):
        super(TinyMPCNet, self).__init__()
        # 4 bemenet -> 16 neuron -> 16 neuron -> 10 kimenet (1 delta + 9 dual változó)
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.network(x)

def train():
    print("Dinamikus adatok betöltése...")
    df = pd.read_csv('data/mpc_dynamic_dataset.csv')
    
    X = df[['e_y', 'e_vy', 'e_psi', 'e_r']].values
    
    # Kinyerjük a 'delta' oszlopot ÉS a 9 darab 'y_...' oszlopot
    target_cols = ['delta'] + [f'y_{i}' for i in range(9)]
    y = df[target_cols].values  # <--- EZ VÁLTOZOTT! Most 10 célváltozónk van.
    
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
    
    # A Mean Squared Error tökéletes ide, mert egyszerre fogja büntetni 
    # a kormányszög és a dual változók tévesztését is!
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005) 
    epochs = 300 
    
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
    print("Tanítás kész! A modell most már a belső állapotokat is megérti.")

if __name__ == "__main__":
    train()