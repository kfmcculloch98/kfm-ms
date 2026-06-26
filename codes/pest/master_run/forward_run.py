import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def run_surrogate():
    print("[Surrogate] Matrix forward wrapper execution initiated...")
    
    # Verify and load the precalculated global response matrix
    if not os.path.exists("G_real_basis.npy"):
        print("[-] Error: 'G_real_basis.npy' missing from active worker context.")
        sys.exit(1)
    G = np.load("G_real_basis.npy")
    
    # Read the parameter tracking file updated by PEST for this worker iteration
    if not os.path.exists("master_truth.csv"):
        print("[-] Error: 'master_truth.csv' missing from active worker context.")
        sys.exit(1)
    q_df = pd.read_csv("master_truth.csv")
    
    # Enforce strict sorting to ensure parameter values align with the matrix G columns
    if "well_id" in q_df.columns and "sp" in q_df.columns:
        q_df = q_df.sort_values(by=["well_id", "sp"]).reset_index(drop=True)
    
    # Fallback to identify what parameter column PEST is currently adjusting
    q_col = 'parval1' if 'parval1' in q_df.columns else 'q'
    q_vec = q_df[q_col].values.astype(float)
    
    #  G * q = b_sim
    try:
        b_sim = G @ q_vec
    except ValueError as e:
        print(f"[-] Matrix multiplication size mismatch error: {str(e)}")
        print(f"    Matrix G shape: {G.shape}, Vector q shape: {q_vec.shape}")
        sys.exit(1)
    
    # Use dummy_obs.csv as the clean structural template
    if not os.path.exists("dummy_obs.csv"):
        print("[-] Error: Template file 'dummy_obs.csv' is missing from worker directory.")
        sys.exit(1)
    obs_df = pd.read_csv("dummy_obs.csv")
    
    # Force template to lower-case to align with pyemu.add_observations() column defaults
    obs_df["obsnme"] = obs_df["obsnme"].astype(str).str.lower()
    
    # Extract the numeric index components using clear text filters
    obs_df['p_num'] = obs_df['obsnme'].str.extract(r'_p(\d+)').astype(int)
    obs_df['c_num'] = obs_df['obsnme'].str.extract(r'_cp(\d+)').astype(int)
    
    # Sort the dataframe layout chronologically to match the matrix G output dimensions exactly
    obs_df = obs_df.sort_values(by=['p_num', 'c_num']).reset_index(drop=True)
    
    # Inject the simulated drawdowns safely into the correctly aligned rows
    obs_df["obsval"] = b_sim
    
    # Clean up temporary tracking columns before export
    obs_df = obs_df.drop(columns=['p_num', 'c_num'])

    # Print a summary of the surrogate execution for diagnostics
    # Print a b_sim summary for verification
    print(f"[Surrogate] Forward solution completed. Total observations: {len(b_sim)}")
    max_drawdown = np.max(b_sim)
    min_drawdown = np.min(b_sim)    
    average_drawdown = np.mean(b_sim)
    print(f"[Surrogate] Drawdown summary: Max={max_drawdown:.4f}, Min={min_drawdown:.4f}, Avg={average_drawdown:.4f}")
    
    # Export and generate 'obs.csv' from scratch for PEST to scrape
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

