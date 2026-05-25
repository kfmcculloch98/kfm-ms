import pandas as pd
import numpy as np
import os

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
