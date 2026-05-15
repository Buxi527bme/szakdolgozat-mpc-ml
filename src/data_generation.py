import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

class KinematicBicycleModel:
    def __init__(self, L=2.5, dt=0.05):
        """
        L: tengelytávolság (méter)
        dt: mintavételi idő / időlépés (másodperc)
        """
        self.L = L
        self.dt = dt
        
        # Állapotvektor: [x, y, v, psi (yaw szög)]
        self.state = np.zeros(4)
        
    def set_state(self, x, y, v, psi):
        """Kezdőállapot beállítása."""
        self.state = np.array([x, y, v, psi], dtype=float)
        
    def update(self, v, delta):
        """
        Állapot frissítése az irányítási bemenetek alapján.
        v: jármű sebessége (m/s) - itt most a sebességet is vezéreljük a kormányszög mellett
        delta: kormányszög (radián)
        """
        x, y, _, psi = self.state
        
        # Differenciálegyenletek a kinematikai modellhez (Euler diszkretizáció)
        x_new = x + v * np.cos(psi) * self.dt
        y_new = y + v * np.sin(psi) * self.dt
        psi_new = psi + (v / self.L) * np.tan(delta) * self.dt
        
        # Új állapot mentése (a sebességet átvesszük a bemenetből)
        self.state = np.array([x_new, y_new, v, psi_new])
        
        return self.state
    

import osqp
import numpy as np
from scipy import sparse

class LateralMPC:
    def __init__(self, N_horizon=10, dt=0.1, L=2.5):
        # Paraméterek
        self.N = N_horizon
        self.dt = dt
        self.L = L
        self.nx = 2  # Állapotok száma: [y, psi]
        self.nu = 1  # Bemenetek száma: [delta (kormányszög)]
        
        # Súlymátrixok beállítása [cite: 187-188]
        # Q: mennyire büntetjük, ha letér a pályáról (y) vagy rossz az orr-irány (psi)
        self.Q = sparse.diags([10.0, 1.0]) 
        # R: mennyire büntetjük a kormányszög hirtelen változását
        self.R = sparse.diags([0.1])
        
        # Kormányszög fizikai korlátai (Amplitúdó kényszer [cite: 226-227])
        # kb. +- 30 fok radiánban kifejezve
        self.delta_min = -0.52
        self.delta_max = 0.52
        
        self.solver = osqp.OSQP()
        self.solver_setup_done = False
        
    def solve(self, current_y, current_psi, target_y, target_psi, v):
        """Kiszámolja az optimális kormányszöget a következő lépésre."""
        
        # 1. Lineáris modell mátrixai (A és B) az adott sebességhez [cite: 151-153]
        A = sparse.csc_matrix([
            [1.0, v * self.dt],
            [0.0, 1.0]
        ])
        B = sparse.csc_matrix([
            [0.0],
            [(v * self.dt) / self.L]
        ])
        
        # 2. QP Költségfüggvény (P és q) felépítése [cite: 252-254]
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.Q,
                               sparse.kron(sparse.eye(self.N), self.R)], format='csc')
        
        target_state = np.array([-target_y, -target_psi])
        q = np.hstack([np.kron(np.ones(self.N), self.Q @ target_state), 
                       self.Q @ target_state, 
                       np.zeros(self.N * self.nu)])
        
        # 3. Dinamikai és kényszer egyenletek felépítése [cite: 223-228]
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), B)
        A_eq = sparse.hstack([Ax, Bu])
        
        leq = np.zeros((self.N + 1) * self.nx)
        leq[0:self.nx] = -np.array([current_y, current_psi])
        ueq = leq
        
        A_ineq = sparse.hstack([sparse.csc_matrix((self.N * self.nu, (self.N + 1) * self.nx)), sparse.eye(self.N * self.nu)])
        lineq = np.ones(self.N * self.nu) * self.delta_min
        uineq = np.ones(self.N * self.nu) * self.delta_max
        
        A_osqp = sparse.vstack([A_eq, A_ineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        
        # 4. OSQP Solver indítása [cite: 259-260]
        if not self.solver_setup_done:
            self.solver.setup(P, q, A_osqp, l, u, warm_start=True, verbose=False)
            self.solver_setup_done = True
        else:
            self.solver.update(q=q, l=l, u=u)
            self.solver.update(Ax=A_osqp.data)
            
        res = self.solver.solve()
        
        # Visszaadjuk a kiszámolt horizontból a legelső kormányszöget [cite: 207-209]
        if res.info.status_val == 1:
            u_opt = res.x[-self.N * self.nu:]
            return u_opt[0]
        else:
            # Ha a solver valamiért nem talál megoldást, vészhelyzeti 0 fok
            return 0.0
        

def generate_double_lane_change(v=5.0, dt=0.1, total_time=10.0):
    """
    Legenerál egy egyszerű dupla sávváltás referenciapályát.
    Visszatér: x_ref, y_ref (numpy tömbök)
    """
    steps = int(total_time / dt)
    x_ref = np.zeros(steps)
    y_ref = np.zeros(steps)
    
    for i in range(steps):
        # Az X pozíció egyenletesen nő a sebesség függvényében
        x_ref[i] = i * v * dt
        
        # Az Y pozíció (sáv) változása
        if x_ref[i] < 10.0:
            y_ref[i] = 0.0              # Eredeti sáv
        elif x_ref[i] < 20.0:
            y_ref[i] = 3.0              # Áttérés a bal sávba (3 méter széles)
        elif x_ref[i] < 30.0:
            y_ref[i] = 3.0              # Haladás a bal sávban
        elif x_ref[i] < 40.0:
            y_ref[i] = 0.0              # Visszatérés a jobb sávba
        else:
            y_ref[i] = 0.0              # Haladás a jobb sávban
            
    # Hogy ne legyenek benne éles, fizikailag lehetetlen ugrások, kicsit "lesimítjuk" egy szűrővel
    from scipy.ndimage import gaussian_filter1d
    y_ref = gaussian_filter1d(y_ref, sigma=3.0)
        
    return x_ref, y_ref

def run_closed_loop_simulation():
    # Szimulációs paraméterek
    v = 5.0         # 5 m/s sebesség
    dt = 0.1        # 0.1s időlépés
    total_time = 12.0
    
    # 1. Referencia pálya generálása
    x_ref, y_ref = generate_double_lane_change(v, dt, total_time)
    steps = len(x_ref)
    
    # Cél orr-irány (psi) kiszámítása a pálya vonalvezetéséből
    psi_ref = np.zeros(steps)
    for i in range(steps - 1):
        dx = x_ref[i+1] - x_ref[i]
        dy = y_ref[i+1] - y_ref[i]
        psi_ref[i] = np.arctan2(dy, dx)
    psi_ref[-1] = psi_ref[-2]
    
    # 2. Rendszerek inicializálása
    car = KinematicBicycleModel(L=2.5, dt=dt)
    car.set_state(x_ref[0], y_ref[0], v, psi_ref[0]) # Az autó a pálya elejéről indul
    
    mpc = LateralMPC(N_horizon=10, dt=dt, L=2.5)
    
    # Adatgyűjtők a rajzoláshoz
    x_history = []
    y_history = []
    
    print("Szimuláció indítása...")
    
    # 3. Szimulációs ciklus (Zárt láncú irányítás)
    # Azért vonjuk ki a horizontot (mpc.N), hogy a pálya legvégén is lásson előre a solver
    for i in range(steps - mpc.N):
        current_x, current_y, current_v, current_psi = car.state
        
        # Célállapot lekérése a pályáról
        target_y = y_ref[i]
        target_psi = psi_ref[i]
        
        # MPC meghívása (Ez itt az online optimalizáció [cite: 199-200])
        delta_opt = mpc.solve(current_y, current_psi, target_y, target_psi, current_v)
        
        # Jármű léptetése a kiszámolt optimális kormányszöggel [cite: 208]
        car.update(v=current_v, delta=delta_opt)
        
        # Adatok mentése
        x_history.append(current_x)
        y_history.append(current_y)
        
    print("Szimuláció vége. Eredmények kirajzolása...")
        
    # 4. Eredmények kirajzolása
    plt.figure(figsize=(12, 5))
    plt.plot(x_ref, y_ref, '--', color='gray', linewidth=2, label='Referencia (Sávváltás)')
    plt.plot(x_history, y_history, '-b', linewidth=2, label='MPC által vezetett autó')
    plt.title('Zárt láncú MPC Szimuláció - ISO Dupla Sávváltás', fontsize=14)
    plt.xlabel('X pozíció (m)')
    plt.ylabel('Y pozíció (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def generate_training_data(num_samples=10000):
    print(f"{num_samples} db minta generálása indul. Ez eltarthat 10-20 másodpercig...")
    
    mpc = LateralMPC(N_horizon=10, dt=0.1, L=2.5)
    v = 5.0 # Fix sebességgel tanítunk
    
    data = []
    
    for i in range(num_samples):
        # Generálunk egy véletlenszerű helyzetet (hibát a célhoz képest)
        # Az autó lehet max 3 méterrel balra/jobbra, és +- 30 fokban elfordulva
        error_y = random.uniform(-3.0, 3.0)
        error_psi = random.uniform(-0.5, 0.5)
        
        # Megkérdezzük az MPC-t, mi a teendő. 
        # A target_y és target_psi most 0.0, mert a hibát az 'error_y' változókban adjuk át
        opt_steering = mpc.solve(current_y=error_y, current_psi=error_psi, 
                                 target_y=0.0, target_psi=0.0, v=v)
        
        # Lementjük a sort: (Y hiba, Szög hiba, Optimális Kormányszög)
        data.append({
            'error_y': error_y,
            'error_psi': error_psi,
            'optimal_delta': opt_steering
        })
        
        # Egy kis visszajelzés, hogy lássuk, nem fagyott le
        if (i + 1) % 2000 == 0:
            print(f"... {i + 1} minta kész")
            
    # Mappa ellenőrzése és fájl mentése
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv('data/mpc_dataset.csv', index=False)
    
    print("Adatgenerálás befejezve! Fájl mentve: data/mpc_dataset.csv")


if __name__ == "__main__":
    # run_closed_loop_simulation() # Ezt most kikommenteltük
    generate_training_data(num_samples=10000)