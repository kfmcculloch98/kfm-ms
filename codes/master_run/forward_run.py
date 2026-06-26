import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def run_surrogate():
    print("[Surrogate] Matrix forward wrapper execution initiated...")
    
    # 1. Verify and load the precalculated global response matrix
    if not os.path.exists("G_real_basis.npy"):
        print("[-] Error: 'G_real_basis.npy' missing from active worker context.")
        sys.exit(1)
    G = np.load("G_real_basis.npy")
    
    # 2. Read the parameter tracking file updated by PEST for this worker iteration
    if not os.path.exists("master_truth.csv"):
        print("[-] Error: 'master_truth.csv' missing from active worker context.")
        sys.exit(1)
    q_df = pd.read_csv("master_truth.csv")
    
    # Enforce strict sorting to ensure parameter values align with your matrix G columns
    if "well_id" in q_df.columns and "sp" in q_df.columns:
        q_df = q_df.sort_values(by=["well_id", "sp"]).reset_index(drop=True)
    
    # Fallback to identify what parameter column PEST is currently adjusting
    q_col = 'parval1' if 'parval1' in q_df.columns else 'q'
    q_vec = q_df[q_col].values.astype(float)
    
    # 3. Fast Forward Matrix Product Step: G * q = b_sim
    try:
        b_sim = G @ q_vec
    except ValueError as e:
        print(f"[-] Matrix multiplication size mismatch error: {str(e)}")
        print(f"    Matrix G shape: {G.shape}, Vector q shape: {q_vec.shape}")
        sys.exit(1)
    
    # 4. FIX: Use dummy_obs.csv as the clean structural template
    if not os.path.exists("dummy_obs.csv"):
        print("[-] Error: Template file 'dummy_obs.csv' is missing from worker directory.")
        sys.exit(1)
    obs_df = pd.read_csv("dummy_obs.csv")
    
    # Inject the simulated drawdowns directly into the dataframe
    obs_df["obsval"] = b_sim
    
    # 5. Export and generate 'obs.csv' from scratch for PEST to scrape
    obs_df.to_csv("obs.csv", index=False)
    print(f"[Surrogate] Success. Generated 'obs.csv' with {len(b_sim)} observation channels.")

def main():

    try:
       os.remove(r'obs.csv')
    except Exception as e:
       print(r'error removing tmp file:obs.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    run_surrogate()

if __name__ == '__main__':
    mp.freeze_support()
    main()

