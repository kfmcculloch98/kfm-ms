import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def run_surrogate():
    # surrogate matrix logic
    G = np.load("G_real_basis.npy") 
    q_df = pd.read_csv("master_truth.csv")
    q_df = q_df.sort_values(by=["sp", "well_id"])
    q_vec = q_df["q"].values 
    b_sim = (G @ q_vec) 
    
    # read in dummy observations data frame
    obs_df = pd.read_csv("dummy_obs.csv") 
    
    # update the values
    obs_df["obsval"] = b_sim
    
    # write back to a filename PEST expects
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

