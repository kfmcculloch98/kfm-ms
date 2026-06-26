import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def run_surrogate():
    """
    Computes rapid surrogate drawdowns at perimeter control points 
    using the pre-calculated block-convolution response matrix (G),
    accounting for PstFrom's internal folder routing.
    """
    print("[Surrogate] Booting forward engine matrix multiplier...")
    
    # 1. Load the pre-calculated global response matrix
    if not os.path.exists("G_real_basis.npy"):
        raise FileNotFoundError("Global response matrix 'G_real_basis.npy' missing from workspace.")
    G = np.load("G_real_basis.npy") 
    
    # 2. Read the parameter mapping table modified by PEST for this iteration
    if not os.path.exists("master_truth.csv"):
        raise FileNotFoundError("Missing parameter mapping file: master_truth.csv")
    q_df = pd.read_csv("master_truth.csv")
    
    # Explicitly sort the data to guarantee strict indexing alignment with matrix columns
    q_df = q_df.sort_values(by=["well_id", "sp"]).reset_index(drop=True)
    q_vec = q_df["q"].values.astype(float) 
    
    # 3. Fast Forward Step Matrix Evaluation (G * q = b_sim)
    b_sim = G @ q_vec 
    
    # 4. Locate the PstFrom nested location of obs.csv dynamically
    # PstFrom typically routes tabular files inside an output directory or multichan folder
    possible_paths = [
        Path("obs.csv"),
        Path("multichan") / "obs.csv",
        Path("output") / "obs.csv"
    ]
    
    target_obs_path = None
    for path in possible_paths:
        if path.exists():
            target_obs_path = path
            break
            
    if target_obs_path is None:
        # Debugging step: list out files to find where PstFrom placed it
        print("[-] Error: Could not locate 'obs.csv' in standard worker paths.")
        print("Current directory structure:")
        for root, dirs, files in os.walk("."):
            for f in files:
                if "obs.csv" in f:
                    print(f" Found potential match at: {os.path.join(root, f)}")
        raise FileNotFoundError("PstFrom structured file 'obs.csv' is missing from worker scope.")

    print(f"[+] Located observation table at: {target_obs_path}")
    obs_df = pd.read_csv(target_obs_path) 
    
    # Map the simulated drawdowns directly into the observation frame
    obs_df["obsval"] = b_sim
    
    # Export the updated data vector back to disk for PEST to scrape
    obs_df.to_csv(target_obs_path, index=False)
    print(f"[Surrogate] Success. Evaluated {len(b_sim)} observation channels.")

if __name__ == "__main__":
    try:
        run_surrogate()
    except Exception as e:
        print(f"[-] FATAL SURROGATE ERROR: {str(e)}")
        sys.exit(1)