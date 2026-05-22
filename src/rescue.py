import numpy as np
import torch
import pickle
import os
import sys

# Hozzáadjuk a src mappát a path-hoz, hogy a train_model import működjön
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import TinyMPCNet

def generate_cpp():
    # Golyóálló útvonalak keresése (egy szinttel feljebb lépünk a src-ből)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'tinympc_ai_weights.pth')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    device = torch.device("cpu")
    model = TinyMPCNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    def print_eigen_vec(name, vec):
        return f"    Eigen::VectorXd {name}({len(vec)});\n    {name} << {', '.join(map(str, vec))};\n"

    def print_eigen_mat(name, mat):
        rows, cols = mat.shape
        return f"    Eigen::MatrixXd {name}({rows}, {cols});\n    {name} << {', '.join(map(str, mat.flatten()))};\n"

    cpp_code = """
    // ====================================================================
    // 🚀 AI WARM-START INICIALIZÁLÁSA (VÉGLEGES, HIBÁTLAN)
    // ====================================================================
    
    // 1. Állapot kinyerése (y, vy, psi, r) a solver memóriájából
    Eigen::VectorXd x_in(4);
    for(int i = 0; i < 4; i++) {
        x_in(i) = solver->work->x(i, 0); 
    }

    // 2. Normalizáció (StandardScaler)
"""
    cpp_code += print_eigen_vec("x_mean", scaler.mean_)
    cpp_code += print_eigen_vec("x_scale", scaler.scale_)
    cpp_code += "    Eigen::VectorXd x_scaled = (x_in - x_mean).cwiseQuotient(x_scale);\n\n"

    cpp_code += "    // 3. A Neurális Hálózat Súlyai\n"
    cpp_code += print_eigen_mat("W1", model.network[0].weight.detach().numpy())
    cpp_code += print_eigen_vec("b1", model.network[0].bias.detach().numpy())
    cpp_code += print_eigen_mat("W2", model.network[2].weight.detach().numpy())
    cpp_code += print_eigen_vec("b2", model.network[2].bias.detach().numpy())
    cpp_code += print_eigen_mat("W3", model.network[4].weight.detach().numpy())
    cpp_code += print_eigen_vec("b3", model.network[4].bias.detach().numpy())

    cpp_code += """
    // 4. Mátrixszorzások és ReLU (Forward Pass)
    Eigen::VectorXd z1 = W1 * x_scaled + b1;
    Eigen::VectorXd a1 = z1.cwiseMax(0.0);

    Eigen::VectorXd z2 = W2 * a1 + b2;
    Eigen::VectorXd a2 = z2.cwiseMax(0.0);

    Eigen::VectorXd ai_output = W3 * a2 + b3;

    // 5. Beillesztés CSAK a legelső indexre
    solver->work->u(0, 0) = ai_output(0);
    std::cout << "[C++] AI Motor fut! Delta: " << ai_output(0) << std::endl;
    // ====================================================================
"""
    print(cpp_code)

if __name__ == "__main__":
    generate_cpp()