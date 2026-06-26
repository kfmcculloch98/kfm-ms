import os
from pathlib import Path
import time
import sys
import numpy as np
import pandas as pd


def run_surrogate():
    base_dir = Path(".").resolve()
    g_path = base_dir / "G_real_basis.npy"
    param_path = base_dir / "p_inst0_grid.csv"
    out_path = base_dir / "sim_drawdown.txt"

    if not g_path.exists() or not param_path.exists():
        sys.exit(1)

    G = np.load(g_path)

    df_pars = None
    for attempt in range(20):
        try:
            df_pars = pd.read_csv(param_path)
            if "parval1" in df_pars.columns:
                break
        except Exception:
            time.sleep(0.05)

    if df_pars is None or "parval1" not in df_pars.columns:
        sys.exit(1)

    q_vec = df_pars["parval1"].to_numpy().flatten()

    if len(q_vec) != G.shape[1]:  # Ensure direct column indexing evaluation
        sys.exit(1)

    b_sim = G @ q_vec
    np.savetxt(out_path, b_sim, fmt="%.6f")
    sys.exit(0)


if __name__ == "__main__":
    run_surrogate()
