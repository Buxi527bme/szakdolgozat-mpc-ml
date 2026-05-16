import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from scipy.ndimage import gaussian_filter1d
import tinympc
from tinympc import TinyMPC  # Biztosítjuk a helyes importot

# --- 1. MODELLEK ---

class DynamicBicycleModel:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.m, self.Iz = 1500.0, 3000.0
        self.lf, self.lr = 1.2, 1.3
        self.Cf, self.Cr = 50000.0, 50000.0
        self.state = np.zeros(6)
        
    def set_state(self, x, y, v_x, v_y, psi, r):
        self.state = np.array([x, y, v_x, v_y, psi, r], dtype=float)
        
    def update(self, delta):
        x, y, v_x, v_y, psi, r = self.state
        v_x = max(v_x, 1.0)
        alpha_f = delta - np.arctan2((v_y + self.lf * r), v_x)
        alpha_r = - np.arctan2((v_y - self.lr * r), v_x)
        F_yf, F_yr = self.Cf * alpha_f, self.Cr * alpha_r
        a_y = (F_yf + F_yr) / self.m - v_x * r
        r_dot = (self.lf * F_yf - self.lr * F_yr) / self.Iz
        self.state = np.array([
            x + (v_x * np.cos(psi) - v_y * np.sin(psi)) * self.dt,
            y + (v_x * np.sin(psi) + v_y * np.cos(psi)) * self.dt,
            v_x,
            v_y + a_y * self.dt,
            psi + r * self.dt,
            r + r_dot * self.dt
        ])
        return self.state

# --- 2. SOLVER (LateralMPC definíciója) ---

class LateralMPC:
    def __init__(self, N_horizon=10, dt=0.1, Q_diag=[30.0, 0.1, 5.0, 0.1], R_diag=[0.1], u_bound=0.52):
        self.N, self.dt = N_horizon, dt
        self.nx, self.nu = 4, 1
        
        # Jármű paraméterek
        m, Iz, lf, lr, Cf, Cr = 1500.0, 3000.0, 1.2, 1.3, 50000.0, 50000.0
        v_x = 10.0
        
        Ac = np.array([
            [0, 1, 0, 0],
            [0, -(2*Cf + 2*Cr)/(m*v_x), 0, -(2*Cf*lf - 2*Cr*lr)/(m*v_x) - v_x],
            [0, 0, 0, 1],
            [0, -(2*Cf*lf - 2*Cr*lr)/(Iz*v_x), 0, -(2*Cf*lf**2 + 2*Cr*lr**2)/(Iz*v_x)]
        ])
        Bc = np.array([[0], [2*Cf/m], [0], [2*Cf*lf/Iz]])
        
        self.Ad = np.eye(4) + Ac * self.dt
        self.Bd = Bc * self.dt
        
        # Szigorított dinamikus paraméterek
        self.Q = Q_diag
        self.R = R_diag
        self.u_min, self.u_max = [-u_bound], [u_bound]
        
        self.x_min = [-20.0, -10.0, -3.14, -5.0]
        self.x_max = [20.0, 10.0, 3.14, 5.0]

        # Példányosítás és Setup
        self.prob = TinyMPC()
        self.prob.setup(self.Ad.astype(np.float64), 
                        self.Bd.astype(np.float64), 
                        np.diag(self.Q).astype(np.float64), 
                        np.diag(self.R).astype(np.float64), 
                        int(self.N), 
                        rho=0.01,
                        x_min=np.array(self.x_min).astype(np.float64), 
                        x_max=np.array(self.x_max).astype(np.float64), 
                        u_min=np.array(self.u_min).astype(np.float64), 
                        u_max=np.array(self.u_max).astype(np.float64))
        
        self.prob.update_settings(abs_pri_tol=1e-4, abs_dua_tol=1e-4, max_iter=1000)

    def solve(self, state_error, warm_start_u=None):
        self.prob.set_x0(state_error.astype(np.float64))
        control_N = self.N - 1
        
        if warm_start_u is not None:
            U_guess = np.tile(warm_start_u, (self.nu, control_N)).astype(np.float64) 
            self.prob.set_warm_start(U_guess)
        else:
            self.prob.set_warm_start(np.zeros((self.nu, control_N)).astype(np.float64))
        
        self.prob.solve()
        res = self.prob.get_solution()
        iters = self.prob.get_info().iter
        return res.u[:, 0][0], iters

# --- 3. UTILS & ADATGENERÁLÁS ---

def generate_double_lane_change(v=10.0, dt=0.1, total_time=15.0):
    steps = int(total_time / dt)
    x_ref, y_ref = np.zeros(steps), np.zeros(steps)
    for i in range(steps):
        x_ref[i] = i * v * dt
        if x_ref[i] < 30.0: y_ref[i] = 0.0
        elif x_ref[i] < 70.0: y_ref[i] = 3.0
        elif x_ref[i] < 110.0: y_ref[i] = 3.0
        elif x_ref[i] < 150.0: y_ref[i] = 0.0
        else: y_ref[i] = 0.0
    return x_ref, gaussian_filter1d(y_ref, sigma=15.0)

def generate_slalom(v=10.0, dt=0.1, total_time=15.0, amplitude=1.2, frequency=0.08):
    """
    Folyamatos szlalom (szinusz) pálya generálása.
    amplitude: a kitérés mértéke méterben.
    frequency: a szlalom sűrűsége (Hz).
    """
    steps = int(total_time / dt)
    t = np.linspace(0, total_time, steps)
    x_ref = v * t
    y_ref = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return x_ref, y_ref

def generate_dynamic_training_data(num_samples=10000):
    print(f"🚀 {num_samples} db dinamikus minta generálása (N=10, u_horizont=9)...")
    mpc = LateralMPC(N_horizon=10, dt=0.1)
    data = []
    
    for i in range(num_samples):
        # Véletlenszerű állapotok a tanításhoz
        # A ciklusban a brutális 3.0 méter helyett reálisabb hibákat kérünk
        err_y = random.uniform(-0.75, 0.75)  # Reális sávtartási hiba
        err_vy = random.uniform(-0.25, 0.25)
        err_psi = random.uniform(-0.1, 0.1)
        err_r = random.uniform(-0.25, 0.25)
        state_err = np.array([err_y, err_vy, err_psi, err_r])
        
        # Megoldás a frissített solve függvénnyel
        delta, iters = mpc.solve(state_err) 
        
        # Csak a konvergált megoldásokat mentjük
        if iters < 950: 
            data.append({
                'e_y': err_y, 'e_vy': err_vy, 
                'e_psi': err_psi, 'e_r': err_r, 
                'delta': delta
            })
        
        if (i+1) % 2000 == 0:
            print(f"... {i+1} minta kész")

    # Adatok mentése fokozott biztonsággal
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(data)
    file_path = 'data/mpc_dynamic_dataset.csv'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        df.to_csv(f, index=False)
        f.flush()
        if hasattr(os, 'fsync'):
            os.fsync(f.fileno())

    print(f"✅ SIKER! {len(df)} minta elmentve: {file_path}")

if __name__ == "__main__":
    generate_dynamic_training_data(num_samples=10000)