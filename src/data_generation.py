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
    def __init__(self, N_horizon=10, dt=0.1, Q_diag=[20.0, 0.1, 4.0, 0.1], R_diag=[0.25], u_bound=0.8):
        self.N, self.dt = N_horizon, dt
        self.nx, self.nu = 4, 1
        
        # Jármű paraméterek
        m, Iz, lf, lr, Cf, Cr = 1500.0, 3000.0, 1.2, 1.3, 50000.0, 50000.0
        v_x = 6.0
        
        Ac = np.array([
            [0, 1, 0, 0],
            [0, -(2*Cf + 2*Cr)/(m*v_x), 0, -(2*Cf*lf - 2*Cr*lr)/(m*v_x) - v_x],
            [0, 0, 0, 1],
            [0, -(2*Cf*lf - 2*Cr*lr)/(Iz*v_x), 0, -(2*Cf*lf**2 + 2*Cr*lr**2)/(Iz*v_x)]
        ])
        Bc = np.array([[0], [2*Cf/m], [0], [2*Cf*lf/Iz]])
        
        self.Ad = np.eye(4) + Ac * self.dt
        self.Bd = Bc * self.dt
        
        self.Q = Q_diag
        self.R = R_diag
        self.u_min, self.u_max = [-u_bound], [u_bound]
        
        self.x_min = [-20.0, -10.0, -3.14, -5.0]
        self.x_max = [20.0, 10.0, 3.14, 5.0]

        self.prob = TinyMPC()
        self.prob.setup(self.Ad.astype(np.float64), 
                        self.Bd.astype(np.float64), 
                        np.diag(self.Q).astype(np.float64), 
                        np.diag(self.R).astype(np.float64), 
                        int(self.N), 
                        rho=0.3,  # <-- stabilabb, mint 0.01
                        x_min=np.array(self.x_min).astype(np.float64), 
                        x_max=np.array(self.x_max).astype(np.float64), 
                        u_min=np.array(self.u_min).astype(np.float64), 
                        u_max=np.array(self.u_max).astype(np.float64))
        
        self.prob.update_settings(abs_pri_tol=1e-3, abs_dua_tol=1e-3, max_iter=1000)

    def solve(self, state_error, warm_start_U=None):
        self.prob.set_x0(state_error.astype(np.float64))
        control_N = self.N - 1
        
        if warm_start_U is None:
            U_guess = np.zeros((self.nu, control_N), dtype=np.float64)
        else:
            U_guess = np.asarray(warm_start_U, dtype=np.float64)
            if U_guess.shape != (self.nu, control_N):
                raise ValueError(f"warm_start_U shape mismatch: {U_guess.shape} != ({self.nu}, {control_N})")
        
        self.prob.set_warm_start(U_guess)
        self.prob.solve()
        res = self.prob.get_solution()
        iters = self.prob.get_info().iter
        return res.u[:, 0][0], iters, res.u

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

def generate_slalom(v=10.0, dt=0.1, total_time=60.0, amplitude=1.2, frequency=0.06):
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

def generate_dynamic_training_data(num_rollouts=10):
    print(f"🚀 Rollout alapú adatgenerálás (N=10)...")
    mpc = LateralMPC(N_horizon=10, dt=0.1)

    v_x, dt = 6.0, 0.1
    x_ref, y_ref = generate_slalom(v_x, dt, total_time=60.0, amplitude=1.2, frequency=0.06)

    psi_ref = np.zeros(len(x_ref))
    for i in range(len(x_ref) - 1):
        psi_ref[i] = np.arctan2(y_ref[i+1] - y_ref[i], x_ref[i+1] - x_ref[i])

    data = []

    for rollout in range(num_rollouts):
        car = DynamicBicycleModel(dt=dt)
        car.set_state(x_ref[0], y_ref[0], v_x, 0.0, psi_ref[0], 0.0)

        for i in range(len(x_ref) - mpc.N):
            _, curr_y, _, curr_vy, curr_psi, curr_r = car.state
            state_err = np.array([
                curr_y - y_ref[i] + np.random.normal(0, 0.05),
                curr_vy + np.random.normal(0, 0.02),
                curr_psi - psi_ref[i] + np.random.normal(0, 0.02),
                curr_r + np.random.normal(0, 0.02)
            ])

            state_err = np.clip(state_err, mpc.x_min, mpc.x_max)

            delta, iters, _ = mpc.solve(state_err)

            data.append({
                'e_y': state_err[0], 'e_vy': state_err[1],
                'e_psi': state_err[2], 'e_r': state_err[3],
                'delta': delta
            })

            car.update(delta)

        print(f"✅ rollout {rollout+1}/{num_rollouts} kész")

    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(data)
    file_path = 'data/mpc_dynamic_dataset.csv'
    df.to_csv(file_path, index=False)
    print(f"✅ SIKER! {len(df)} minta elmentve: {file_path}")

if __name__ == "__main__":
    generate_dynamic_training_data(num_rollouts=10)