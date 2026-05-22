import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from scipy.ndimage import gaussian_filter1d
import tinympc
from tinympc import TinyMPC 
from config_loader import load_config
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
class LateralMPC:
    def __init__(
        self,
        N_horizon=10,
        dt=0.1,
        v_x=6.0,
        Q_diag=[20.0, 0.1, 4.0, 0.1],
        R_diag=[0.25],
        u_bound=0.8,
        rho=0.3,
        abs_pri_tol=1e-3,
        abs_dua_tol=1e-3,
        x_min=[-50.0, -20.0, -6.28, -8.0],
        x_max=[50.0, 20.0, 6.28, 8.0],
    ):
        self.N, self.dt = N_horizon, dt
        self.nx, self.nu = 4, 1
        
        m, Iz, lf, lr, Cf, Cr = 1500.0, 3000.0, 1.2, 1.3, 50000.0, 50000.0
        
        Ac = np.array([
            [0, 1, 0, 0],
            [0, -(2*Cf + 2*Cr)/(m*v_x), 0, -(2*Cf*lf - 2*Cr*lr)/(m*v_x) - v_x],
            [0, 0, 0, 1],
            [0, -(2*Cf*lf - 2*Cr*lr)/(Iz*v_x), 0, -(2*Cf*lf**2 + 2*Cr*lr**2)/(Iz*v_x)]
        ])
        Bc = np.array([[0], [2*Cf/m], [0], [2*Cf*lf/Iz]])
        
        self.Ad = np.eye(4) + Ac * self.dt
        self.Bd = Bc * self.dt
        
        self.Q = list(Q_diag)
        self.R = list(R_diag)
        self.u_min, self.u_max = [-u_bound], [u_bound]
        
        self.x_min = list(x_min)
        self.x_max = list(x_max)

        self.prob = TinyMPC()
        self.prob.setup(self.Ad.astype(np.float64), 
                        self.Bd.astype(np.float64), 
                        np.diag(self.Q).astype(np.float64), 
                        np.diag(self.R).astype(np.float64), 
                        int(self.N), 
                        rho=rho,  # <-- stabilabb, mint 0.01
                        x_min=np.array(self.x_min).astype(np.float64), 
                        x_max=np.array(self.x_max).astype(np.float64), 
                        u_min=np.array(self.u_min).astype(np.float64), 
                        u_max=np.array(self.u_max).astype(np.float64))
        
        self.prob.update_settings(abs_pri_tol=abs_pri_tol, abs_dua_tol=abs_dua_tol, max_iter=1000)

    def solve(self, state_error, warm_start_U=None, warm_start_Y=None):
        self.prob.set_x0(state_error.astype(np.float64))
        control_N = self.N - 1
        
        # 1. Primal (Kormányszög) melegindítás
        if warm_start_U is None:
            U_guess = np.zeros((self.nu, control_N), dtype=np.float64)
        else:
            U_guess = np.asarray(warm_start_U, dtype=np.float64)
            if U_guess.shape != (self.nu, control_N):
                raise ValueError(f"warm_start_U shape mismatch: {U_guess.shape} != ({self.nu}, {control_N})")
        
        self.prob.set_warm_start(U_guess)

        # 2. Dual (Lagrange-szorzók) melegindítás - EZT FOGJA HASZNÁLNI AZ AI!
        if warm_start_Y is not None:
            Y_guess = np.asarray(warm_start_Y, dtype=np.float64)
            self.prob.set_warm_start_dual(Y_guess)

        # Megoldás
        self.prob.solve()
        res = self.prob.get_solution()
        iters = self.prob.get_info().iter
        
        # 3. Kiolvassuk a memóriából a tökéletes dual állapotokat az adatgeneráláshoz
        y_matrix = self.prob.get_y()

        # Visszaadjuk az y_matrix-ot is a negyedik helyen!
        return res.u[:, 0][0], iters, res.u, y_matrix
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
    Folyamatos szlalom pálya generálása.
    amplitude: a kitérés mértéke méterben.
    frequency: a szlalom sűrűsége (Hz).
    """
    steps = int(total_time / dt)
    t = np.linspace(0, total_time, steps)
    x_ref = v * t
    y_ref = amplitude * np.sin(2 * np.pi * frequency * t)
    y_ref[-20:] = np.linspace(y_ref[-20], 0, 20)
    
    return x_ref, y_ref

def generate_dynamic_training_data(num_rollouts=10):
    cfg = load_config()
    print(f"Rollout alapú adatgenerálás (N={cfg['N']})...")
    mpc = LateralMPC(
        N_horizon=cfg["N"],
        dt=cfg["dt"],
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

    v_x, dt = cfg["v_x"], cfg["dt"]
    x_ref, y_ref = generate_slalom(
        v_x,
        dt,
        total_time=cfg["total_time"],
        amplitude=cfg["amplitude"],
        frequency=cfg["frequency"],
    )

    psi_ref = np.zeros(len(x_ref))
    for i in range(len(x_ref) - 1):
        psi_ref[i] = np.arctan2(y_ref[i+1] - y_ref[i], x_ref[i+1] - x_ref[i])

    data = []

    for rollout in range(num_rollouts):
        car = DynamicBicycleModel(dt=dt)
        car.set_state(x_ref[0], y_ref[0], v_x, 0.0, psi_ref[0], 0.0)

        # EZ A BELSŐ CIKLUS HIÁNYZOTT:
        for i in range(len(x_ref) - mpc.N):
            _, curr_y, _, curr_vy, curr_psi, curr_r = car.state
            
            # Kiszámoljuk az állapothibát (ez is eltűnt a képeden)
            state_err = np.array([
                curr_y - y_ref[i] + np.random.normal(0, 0.02),
                curr_vy + np.random.normal(0, 0.01),
                curr_psi - psi_ref[i] + np.random.normal(0, 0.01),
                curr_r + np.random.normal(0, 0.01)
            ])

            state_err = np.clip(state_err, mpc.x_min, mpc.x_max)

            # Most már 4 változót várunk a solve-tól!
            delta, iters, _, y_matrix = mpc.solve(state_err)

            # Kilapítjuk a mátrixot (4x10 -> 40 elemű vektor)
            y_flat = np.array(y_matrix).flatten()

            row_data = {
                'e_y': state_err[0], 'e_vy': state_err[1],
                'e_psi': state_err[2], 'e_r': state_err[3],
                'delta': delta
            }

            # Dinamikusan hozzáadjuk mind a 40 y értéket (y_0, y_1 ... y_39)
            for j, y_val in enumerate(y_flat):
                row_data[f'y_{j}'] = y_val

            data.append(row_data)

            car.update(delta)

        print(f"rollout {rollout+1}/{num_rollouts} kész")

    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(data)
    file_path = 'data/mpc_dynamic_dataset.csv'
    df.to_csv(file_path, index=False)
    print(f"Siker! {len(df)} minta elmentve: {file_path}")

if __name__ == "__main__":
    generate_dynamic_training_data(num_rollouts=15)
