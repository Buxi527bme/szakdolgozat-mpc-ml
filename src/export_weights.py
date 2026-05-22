import torch
from train_model import TinyMPCNet
import numpy as np

# Modell betöltése
device = torch.device("cpu")
model = TinyMPCNet()
model.load_state_dict(torch.load('models/tinympc_ai_weights.pth', map_location=device, weights_only=True))
model.eval()

def export_to_header(model, filename="src/model_weights.h"):
    with open(filename, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <vector>\n\n")
        
        # Súlyok kinyerése
        for i, layer in enumerate([l for l in model.network if hasattr(l, 'weight')]):
            w = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            
            f.write(f"// Layer {i} weights and biases\n")
            f.write(f"const std::vector<float> W{i} = {{")
            f.write(", ".join(map(str, w.flatten())))
            f.write("};\n\n")
            
            f.write(f"const std::vector<float> B{i} = {{")
            f.write(", ".join(map(str, b.flatten())))
            f.write("};\n\n")
            
    print(f"Súlyok sikeresen exportálva ebbe: {filename}")

export_to_header(model)