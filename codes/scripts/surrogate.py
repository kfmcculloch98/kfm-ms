# surrogate.py
import pandas as pd
import numpy as np
import os

def run_surrogate():
    # 1. Your math logic
    G = np.load("G_real_basis.npy") 
    q_df = pd.read_csv("master_truth.csv")
    q_df = q_df.sort_values(by=["sp", "well_id"])
    q_vec = q_df["q"].values 
    b_sim = (G @ q_vec) 
    
    # 2. Read from the "safe" backup that wasn't deleted
    obs_df = pd.read_csv("dummy_obs.csv") 
    
    # 3. Update the values
    obs_df["obsval"] = b_sim
    
    # 4. Write back to the filename PEST expects
    obs_df.to_csv("obs.csv", index=False)
