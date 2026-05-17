import argparse
import itertools
import json
import os
import subprocess
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OVERRIDE_PATH = os.path.join(ROOT_DIR, "src", "config_override.json")
METRICS_PATH = os.path.join(ROOT_DIR, "metrics.json") # Ide várjuk a szimuláció eredményét

def _load_grid(grid_path=None):
    if grid_path is None:
        return {}
    with open(grid_path, "r", encoding="utf-8") as f:
        grid = json.load(f) or {}
    if not isinstance(grid, dict):
        raise ValueError("Grid JSON must be a dictionary of key -> list.")
    for key, values in grid.items():
        if not isinstance(values, list):
            raise ValueError(f"Grid key '{key}' must map to a list.")
    return grid

def _iter_overrides(grid):
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))

def _write_override(override):
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(override, f, indent=2)

def _run_pipeline():
    scripts = ["src/data_generation.py", "src/train_model.py", "src/simulation.py"]
    for script in scripts:
        subprocess.run([sys.executable, script], cwd=ROOT_DIR, check=True)

def run_sweep(grid_path=None):
    grid = _load_grid(grid_path)
    master_results = []
    
    try:
        for run_idx, override in enumerate(_iter_overrides(grid), start=1):
            print(f"\n🔁 Sweep run {run_idx}: {override}")
            _write_override(override)
            
            # Töröljük a régi metrikát, hogy ne olvassunk be fals adatot
            if os.path.exists(METRICS_PATH):
                os.remove(METRICS_PATH)
                
            _run_pipeline()
            
            # Sikeres futás után beolvassuk a friss metrikákat
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                
                # Összefésüljük a paramétereket az eredményekkel
                run_summary = {**override, **metrics}
                master_results.append(run_summary)
                
                # CSV frissítése minden iteráció után (biztonsági mentés)
                df = pd.DataFrame(master_results)
                df.to_csv("grid_master_summary.csv", index=False)
            else:
                print(f"⚠️ HIBA: A simulation.py nem hozta létre a metrics.json-t a {run_idx}. körben!")
                
    finally:
        _write_override({})
        print("\n🏆 Grid Search befejezve! Eredmények: grid_master_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run config sweep for MPC pipeline.")
    parser.add_argument(
        "--grid-json",
        default=None,
        help="Path to a JSON file mapping config keys to candidate value lists.",
    )
    args = parser.parse_args()
    run_sweep(args.grid_json)