import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import time
from data_generation import DynamicBicycleModel, generate_double_lane_change, LateralMPC, generate_slalom
from train_model import TinyMPCNet

def run_ultimate_benchmark():
    print("🚀 Ultimate TinyMPC + NumPy AI Warm-start Benchmark indítása...")
    
    # 1. AI Modell betöltése
    device = torch.device("cpu") # Most szándékosan CPU-t használunk a memóriamásolás elkerülésére
    model = TinyMPCNet().to(device)
    model.load_state_dict(torch.load('models/tinympc_ai_weights.pth', map_location=device, weights_only=True))
    model.eval()
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # --- A NUMPY TRÜKK: Súlyok kimentése a villámgyors inferenciához ---
    w1 = model.network[0].weight.detach().numpy()
    b1 = model.network[0].bias.detach().numpy()
    w2 = model.network[2].weight.detach().numpy()
    b2 = model.network[2].bias.detach().numpy()
    w3 = model.network[4].weight.detach().numpy()
    b3 = model.network[4].bias.detach().numpy()

    def fast_numpy_inference(x_scaled):
        # 3 apró mátrixszorzás PyTorch overhead nélkül (< 0.05 ms)
        x = np.maximum(0, x_scaled @ w1.T + b1) # ReLU 1
        x = np.maximum(0, x @ w2.T + b2)        # ReLU 2
        return (x @ w3.T + b3)[0]               # Kimenet

    # 2. Szimulációs környezet
    v_x, dt = 6.0, 0.1
    x_ref, y_ref = generate_slalom(v_x, dt, total_time=60.0, amplitude=1.2, frequency=0.06)
    
    psi_ref = np.zeros(len(x_ref))
    for i in range(len(x_ref)-1):
        psi_ref[i] = np.arctan2(y_ref[i+1]-y_ref[i], x_ref[i+1]-x_ref[i])
    
    car = DynamicBicycleModel(dt=dt)
    car.set_state(x_ref[0], y_ref[0], v_x, 0.0, psi_ref[0], 0.0)
    mpc = LateralMPC(N_horizon=10, dt=dt)
    
    results = {
        'cold_iters': [], 'warm_iters': [], 
        'cold_times_ms': [], 'warm_total_times_ms': []
    }

    print("🏁 Szimuláció futtatása a pályán...")

    prev_U = None

    for i in range(len(x_ref) - 10):
        _, curr_y, _, curr_vy, curr_psi, curr_r = car.state
        state_err = np.array([curr_y - y_ref[i], curr_vy - 0.0, curr_psi - psi_ref[i], curr_r - 0.0])
        state_err = np.clip(state_err, mpc.x_min, mpc.x_max)
        
        # --- A) HIDEGINDÍTÁS ---
        t_start_cold = time.perf_counter()
        _, iter_cold, _ = mpc.solve(state_err, warm_start_U=None)
        results['cold_times_ms'].append((time.perf_counter() - t_start_cold) * 1000)
        results['cold_iters'].append(iter_cold)
        
        # --- B) MELEGINDÍTÁS ---
        t_start_warm = time.perf_counter()
        in_scaled = scaler.transform([state_err])[0]
        ai_delta = fast_numpy_inference(in_scaled)
        
        if prev_U is None:
            U_warm = np.zeros((mpc.nu, mpc.N - 1))
        else:
            # előző optimális U eltolása
            U_warm = np.hstack([prev_U[:, 1:], prev_U[:, -1:]])
        
        # AI tipp az első elemre
        U_warm[0, 0] = ai_delta
        
        delta_warm, iter_warm, U_sol = mpc.solve(state_err, warm_start_U=U_warm)
        prev_U = U_sol
        
        results['warm_total_times_ms'].append((time.perf_counter() - t_start_warm) * 1000)
        results['warm_iters'].append(iter_warm)
        
        if iter_cold == 1000 or iter_warm == 1000:
            print(f"⚠️ FIGYELEM: Infeasible állapot a {i}. lépésnél! A jármű lesodródott.")
        
        car.update(delta_warm)

    # 3. Kiértékelés (első 5 lépés levágása a tranziens miatt)
    f_cold_iter = results['cold_iters'][5:]
    f_warm_iter = results['warm_iters'][5:]
    f_cold_time = results['cold_times_ms'][5:]
    f_warm_time = results['warm_total_times_ms'][5:]

    avg_c_iter, avg_w_iter = np.mean(f_cold_iter), np.mean(f_warm_iter)
    avg_c_time, avg_w_time = np.mean(f_cold_time), np.mean(f_warm_time)

    print(f"\n🏆 --- VÉGSŐ TINYMPC BENCHMARK EREDMÉNYEK ---")
    print(f"Átlagos iteráció (Hideg): {avg_c_iter:.1f}")
    print(f"Átlagos iteráció (Meleg): {avg_w_iter:.1f}  --> JAVULÁS: {((avg_c_iter-avg_w_iter)/avg_c_iter)*100:.1f}%")
    print(f"Átlagos Futási Idő (Hideg): {avg_c_time:.3f} ms")
    print(f"Átlagos Futási Idő (NumPy AI + Meleg): {avg_w_time:.3f} ms --> JAVULÁS: {((avg_c_time-avg_w_time)/avg_c_time)*100:.1f}%")

    # Grafikonok rajzolása a szakdolgozathoz
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(f_cold_iter, label='Hidegindítás', color='red', alpha=0.6)
    plt.plot(f_warm_iter, label='AI Melegindítás', color='green', linewidth=2)
    plt.title('TinyMPC ADMM Iterációszámok')
    plt.ylabel('Iterációk (k)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot([f_cold_time, f_warm_time], labels=['Hidegindítás', 'AI + Melegindítás'])
    plt.title('Valós Futási Idők (ms)')
    plt.ylabel('Idő (ms)')
    plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('eredmeny_tinympc_vegszo.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_ultimate_benchmark()