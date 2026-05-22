import pickle
import numpy as np

# IDE MÁSOLD BE AZ ELŐBB KIMÁSOLT TELJES ÚTVONALAT!
teljes_utvonal = "/Users/Buxi/Documents/Szakdolgozat_MPC_ML/models/scaler.pkl" 

with open(teljes_utvonal, 'rb') as f:
    scaler = pickle.load(f)

print("Ezeket másold be a C++ kódba a mean_x tömbbe:")
print(np.array2string(scaler.mean_, separator=', '))

print("\nEzeket másold be a C++ kódba a scale_x tömbbe:")
print(np.array2string(scaler.scale_, separator=', '))