import torch
import numpy as np

# Feltételezzük, hogy a modelled egy egyszerű MLP:
# Bemenet (4 állapot) -> Rejtett réteg (32 neuron, ReLU) -> Kimenet (9 kontroll lépés)
def export_to_c_header(model_path, output_header_path):
    # Modell betöltése (ha egyedi osztályod van, azt példányosítsd előtte)
    model = torch.load(model_path)
    model.eval()
    
    state_dict = model.state_dict()
    
    with open(output_header_path, 'w') as f:
        f.write("#ifndef TINYMPC_NN_WEIGHTS_H\n")
        f.write("#define TINYMPC_NN_WEIGHTS_H\n\n")
        
        # Végigmegyünk a rétegeken
        layer_idx = 1
        for name, param in state_dict.items():
            param_np = param.detach().cpu().numpy()
            
            # Mátrix kiterítése 1D tömbbé (C-kompatibilis sorfolytonos tárolás)
            flat_data = param_np.flatten()
            array_str = ", ".join([f"{val:.6f}f" for val in flat_data])
            
            if "weight" in name:
                rows, cols = param_np.shape
                f.write(f"// Layer {layer_idx} Weights: {rows}x{cols}\n")
                f.write(f"const float W{layer_idx}[{rows * cols}] = {{{array_str}}};\n\n")
            
            elif "bias" in name:
                size = param_np.shape[0]
                f.write(f"// Layer {layer_idx} Biases: {size}\n")
                f.write(f"const float b{layer_idx}[{size}] = {{{array_str}}};\n\n")
                layer_idx += 1
                
        f.write("#endif // TINYMPC_NN_WEIGHTS_H\n")
        
    print(f"Sikeres exportálás: {output_header_path}")

# Futtatás:
# export_to_c_header('betanitott_tinympc_net.pth', 'weights.h')