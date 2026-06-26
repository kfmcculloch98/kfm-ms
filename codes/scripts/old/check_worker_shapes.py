import os
import numpy as np
import pandas as pd
from pathlib import Path

pest_dir = Path(r"C:\Python\Personal\kfm-ms\codes\pest")
g_path = pest_dir / "G_real_basis.npy"
par_path = pest_dir / "q_inst0_grid.csv"

print("--- ACTIVE PEST SPACE METRICS ---")
if g_path.exists():
    G = np.load(g_path, mmap_mode='r')
    print(f"[+] G_real_basis.npy shape: {G.shape}")
else:
    print("[!] G_real_basis.npy is MISSING from the pest directory!")

if par_path.exists():
    df = pd.read_csv(par_path)
    print(f"[+] q_inst0_grid.csv shape: {df.shape}")
    print(f"[+] Available columns: {list(df.columns)}")
else:
    # Try looking for your parameter source seeding file
    rates_path = pest_dir / "pumping_rates.csv"
    if rates_path.exists():
        df = pd.read_csv(rates_path)
        print(f"[+] pumping_rates.csv shape: {df.shape}")
