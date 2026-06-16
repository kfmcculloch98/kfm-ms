import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def run_surrogate():
    # load the surrogate basis matrix G (precomputed from the full forward model)
    G = np.load("G_real_basis.npy") 
    
    # read the master truth pumping schedule (the "q" vector) from the CSV file
    q_df = pd.read_csv("master_truth.csv")
    
    # strip the "q" column to get the pumping schedule as a numpy array
    q_vec = q_df["q"].values 
    
    # compute the surrogate-predicted drawdown at the control points using matrix multiplication
    b_sim = G @ q_vec 
    
    # read the registered observation structure layout
    obs_df = pd.read_csv("dummy_obs.csv") 
    
    # map the simulated drawdowns directly to the observation target vector
    obs_df["obsval"] = b_sim
    
    # export the final data back to disk where PEST expects it
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

