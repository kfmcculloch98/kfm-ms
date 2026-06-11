import pandas as pd
import numpy as np
import os

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