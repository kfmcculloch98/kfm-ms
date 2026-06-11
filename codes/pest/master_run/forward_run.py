import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def run_surrogate():
    # 1. Load the surrogate matrix operator
    G = np.load("G_real_basis.npy") 
    
    # 2. Read parameters exactly in the native sequence PEST writes them
    q_df = pd.read_csv("master_truth.csv")
    
    # FIXED: Stripped the .sort_values() call to maintain strict 1:1 parameter index matching
    q_vec = q_df["q"].values 
    
    # 3. Perform index-aligned linear convolution matrix multiplication
    b_sim = G @ q_vec 
    
    # 4. Read the registered observation structure layout
    obs_df = pd.read_csv("dummy_obs.csv") 
    
    # 5. Map the simulated drawdowns directly to the observation target vector
    obs_df["obsval"] = b_sim
    
    # 6. Export the final data back to disk where PEST expects it
    obs_df.to_csv("obs.csv", index=False)
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

