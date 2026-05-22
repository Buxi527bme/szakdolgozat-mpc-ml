import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from data_generation import DynamicBicycleModel, LateralMPC, generate_slalom
from config_loader import load_config
# FIGYELD MEG: Nincs több torch, pickle, és train_model import! A Python teljesen "buta" lett.

def run_ultimate_benchmark():
    print("TinyMPC + NATIVE C++ AI Benchmark indítása...")
    cfg = load_config()

    v_x, dt = cfg["v_x"], cfg["dt"]
    x_ref, y_ref = generate_slalom(
        v_x,
        dt,
        total_time=cfg["total_time"],
        amplitude=cfg["amplitude"],
        frequency=cfg["frequency"],
    )
    
    psi_ref = np.zeros(len(x_ref))
    for i in range(len(x_ref)-1):
        psi_ref[i] = np.arctan2(y_ref[i+1]-y_ref[i], x_ref[i+1]-x_ref[i])
    
    car = DynamicBicycleModel(dt=dt)
    car.set_state(x_ref[0], y_ref[0], v_x, 0.0, psi_ref[0], 0.0)
    
    mpc = LateralMPC(
        N_horizon=cfg["N"],
        dt=dt,
        v_x=cfg["v_x"],
        Q_diag=cfg["Q_diag"],
        R_diag=cfg["R_diag"],
        u_bound=cfg["u_bound"],
        rho=cfg["rho"],
        abs_pri_tol=cfg["abs_pri_tol"],
        abs_dua_tol=cfg["abs_dua_tol"],
        x_min=cfg["x_min"],
        x_max=cfg["x_max"],
    )
    
    results = {'iters': [], 'times_ms': []}

    print("Szimuláció futtatása (Tiszta C++ Motorral)")

    print("Szimuláció futtatása (Tiszta C++ Motorral)")

    prev_U = None  # 1. LÉPÉS: Létrehozzuk a memóriaváltozót a ciklus előtt

    for i in range(len(x_ref) - mpc.N):
        _, curr_y, _, curr_vy, curr_psi, curr_r = car.state
        state_err = np.array([curr_y - y_ref[i], curr_vy - 0.0, curr_psi - psi_ref[i], curr_r - 0.0])
        state_err = np.clip(state_err, mpc.x_min, mpc.x_max)
        
        t_start = time.perf_counter()

        if prev_U is None:
            U_warm = np.zeros((mpc.nu, mpc.N - 1), dtype=np.float64, order='F')
        else:
            # A Python csúsztatja a memóriát (ez nagyon gyors, nincs overhead)
            U_warm = np.hstack([prev_U[:, 1:], prev_U[:, -1:]])
            # Szigorú memóriakiosztás a C++ wrapper számára
            U_warm = np.array(U_warm, dtype=np.float64, copy=True, order='F')

        t_start = time.perf_counter()
        
        # A Python átadja az elcsúsztatott memóriát, a C++ megkapja, szinkronizál, és RÁRAKJA az AI-t!
        delta, iters, U_sol = mpc.solve(state_err, warm_start_U=U_warm)
        
        prev_U = U_sol # Ezt már csak elmentjük a következő lépésre
        
        run_time_ms = (time.perf_counter() - t_start) * 1000
        
        results['times_ms'].append(run_time_ms)
        results['iters'].append(iters)
        
        if iters == 1000:
            print(f"Infeasible állapot a {i}. lépésnél!")
        
        car.update(delta)

    # Az első pár lépést (tranziens) levágjuk a mérésből
    f_iters = results['iters'][5:]
    f_times = results['times_ms'][5:]

    avg_iter = np.mean(f_iters)
    avg_time = np.mean(f_times)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n--- TISZTA NATIVE C++ AI EREDMÉNYEK ---")
    print(f"Átlagos iteráció (C++ AI): {avg_iter:.1f}")
    print(f"Átlagos Futási Idő (C++ AI): {avg_time:.3f} ms")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(f_iters, label='Native C++ AI Melegindítás', color='green', linewidth=2)
    plt.title('TinyMPC ADMM Iterációszámok')
    plt.ylabel('Iterációk (k)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot([f_times], labels=['Native C++ AI'])
    plt.title('Valós Futási Idők (ms)')
    plt.ylabel('Idő (ms)')
    plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)  
    plt.close('all')

if __name__ == "__main__":
    run_ultimate_benchmark()