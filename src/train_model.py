import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

# --- 1. A Neurális Háló Architektúrája ---
class TinyMPCNet(nn.Module):
    def __init__(self):
        super(TinyMPCNet, self).__init__()
        # 2 bemenet (y_hiba, psi_hiba) -> 16 neuron -> 16 neuron -> 1 kimenet (kormányszög)
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

def train():
    print("Adatok betöltése és előkészítése...")
    
    # 2. Adatok beolvasása
    df = pd.read_csv('data/mpc_dataset.csv')
    
    # Bemenetek (X) és Kimenet (y) szétválasztása
    X = df[['error_y', 'error_psi']].values
    y = df[['optimal_delta']].values
    
    # 3. Adatok felosztása (80% tanítás, 20% tesztelés)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Normalizálás (Skálázás)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Elmentjük a skálázót, mert a szimulációnál a friss adatokat is ezzel kell majd transzformálni!
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
        
    # PyTorch Tensorokká alakítás
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 5. Eszköz kiválasztása (Kihasználjuk az M1 Pro-t!)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Tanítás ezen az eszközön: {device}")
    
    model = TinyMPCNet().to(device)
    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)
    
    # 6. Tanítási paraméterek
    criterion = nn.MSELoss() # Átlagos négyzetes hiba
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimalizáló
    epochs = 200
    
    print("Tanítás indítása...")
    for epoch in range(epochs):
        model.train()
        
        # Előre terjesztés
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        
        # Visszaterjesztés és súlyok frissítése
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 10 lépésenként kiírjuk, hol tartunk
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_tensor)
                test_loss = criterion(test_preds, y_test_tensor)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")
            
    # 7. Kész modell mentése
    torch.save(model.state_dict(), 'models/tinympc_ai_weights.pth')
    print("Tanítás kész! Modell és Skálázó elmentve a 'models/' mappába.")

if __name__ == "__main__":
    train()