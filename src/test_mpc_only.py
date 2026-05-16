import numpy as np
import time
from data_generation import LateralMPC

def run_mpc_unit_test():
    print("🔧 Izolált TinyMPC Matematikai Teszt Indítása...\n")
    
    # Példányosítjuk az MPC-t (szimulációs háló és AI nélkül)
    try:
        mpc = LateralMPC(N_horizon=10, dt=0.1)
        print("✅ MPC példányosítva a data_generation.py alapján.")
    except Exception as e:
        print(f"❌ Hiba az MPC inicializálásakor: {e}")
        return

    # Tesztesetek: [e_y (méter), e_vy (m/s), e_psi (radián), e_r (rad/s)]
    test_cases = [
        {"name": "1. Enyhe hiba (10 cm letérés)", "state": np.array([0.1, 0.0, 0.0, 0.0])},
        {"name": "2. Közepes hiba (1 méter letérés)", "state": np.array([1.0, 0.0, 0.0, 0.0])},
        {"name": "3. Tiszta szöghiba (kb. 5 fok eltérés)", "state": np.array([0.0, 0.0, 0.1, 0.0])},
        {"name": "4. Kombinált extrém hiba (3m letérés, sodródás)", "state": np.array([3.0, 0.5, 0.2, 0.1])}
    ]
    
    for case in test_cases:
        print(f"▶ Teszt: {case['name']}")
        print(f"  Bemenet (állapothiba): {case['state']}")
        
        start_time = time.time()
        # Szigorúan hidegindítás, melegítő AI tipp nélkül
        try:
            delta, iters = mpc.solve(case['state'], warm_start_u=None)
            calc_time = (time.time() - start_time) * 1000
            
            print(f"  Eredmény -> Kormányszög (delta): {delta:.4f} rad")
            print(f"  Iterációk száma: {iters} (Futási idő: {calc_time:.2f} ms)")
            
            if iters >= 950:
                print("  ❌ HIBA: A solver nem konvergált! (Beakadt a plafonba)")
            else:
                print("  ✅ SIKER: A solver matematikailag stabil ezen a ponton.")
        except Exception as e:
            print(f"  ❌ KRITIKUS HIBA a számolás közben: {e}")
        print("-" * 50)

if __name__ == "__main__":
    run_mpc_unit_test()