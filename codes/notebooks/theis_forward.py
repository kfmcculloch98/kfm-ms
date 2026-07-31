import numpy as np
import scipy.special as sp
import pandas as pd

# Load the file after PEST's multiplier has been applied to it
df_par = pd.read_csv("params.txt", delim_whitespace=True, index_col="parnme")
Q = df_par.loc["Q", "parval"]
t_start = df_par.loc["t_start", "parval"]

T = 0.005 * 1477.0 
S = 1e-5 * 1477.0
r = 500.0
time_array = np.linspace(1.0, 364.0, 52)

s_out = []
for t in time_array:
    if t <= t_start:
        s_out.append(0.0)
    else:
        dt = t - t_start
        u = (r**2 * S) / (4.0 * T * dt)
        s_out.append((Q / (4.0 * np.pi * T)) * sp.exp1(u))

# Write tabular output for PEST tracking
df_obs = pd.DataFrame({
    "obsnme": [f"s_{i:02d}" for i in range(52)],
    "simval": s_out
})
df_obs.to_csv("drawdown.txt", sep=" ", index=False)
