import os
import pandas as pd
import numpy as np
import flopy
from pathlib import Path

# ==============================================================================
# CONFIGURATION AND WORKSPACE SETUP
# ==============================================================================
script_dir = Path(__file__).parent
notebook_dir = os.getcwd()

MODFLOW_PATH = "../binaries/MODFLOW6/windows"
mf6_exe = os.path.join(notebook_dir, MODFLOW_PATH, "mf6.exe")

workspace_base = os.path.join("..", "sims")
root = input("Enter root name of the run to copy boundaries from: ").strip()

# Set up a new workspace for this advanced donut simulation run
donut_ws = os.path.abspath(os.path.join(workspace_base, f"{root}_donut_ts"))
os.makedirs(donut_ws, exist_ok=True)

# Load the exported 0-indexed absolute head data
results_path = script_dir.parent / "results"
chd_file_path = results_path / f"{root}_chd_portable_boundaries.csv"

if not chd_file_path.exists():
    raise FileNotFoundError(f"Could not find boundary file at: {chd_file_path}")

chd_df = pd.read_csv(chd_file_path)

# Synchronized Time Horizon Discretization (Matching Rectangular 70-Week Scope)
nper_pump = 52       # Active pumping timeframe contained in the CSV
nper_sim = 70        # Total simulation horizon (52 active + 18 recovery tracking)
nper = nper_sim      # Point MODFLOW configurations to the full 70-week window

perlen = 7.0    
nstp = 5        
tsmult = 1.0    
tdis_rc = [(perlen, nstp, tsmult) for _ in range(nper_sim)]

# Spatial grid definition
cell_dim, ncol, nrow = 100, 92, 116 

# ==============================================================================
# 1. CONSTRUCT THE DONUT IDOMAIN (ACTIVE OUTSIDE & ON PERIMETER, NO-FLOW INSIDE)
# ==============================================================================
print("Constructing donut domain active mask...")
idomain = np.ones((1, nrow, ncol), dtype=int)

r_min, r_max = chd_df["row"].min(), chd_df["row"].max()
c_min, c_max = chd_df["col"].min(), chd_df["col"].max()

# Deactivate cells strictly inside the boundary rectangle
idomain[0, r_min + 1 : r_max, c_min + 1 : c_max] = 0

# ==============================================================================
# 2. DESIGN THE TIME-SERIES BOUNDARY POINTER LAYOUT (TS MECHANICS)
# ==============================================================================
print("Structuring continuous Time-Series records for the CHD solver...")

# Identify unique boundary nodes to establish static geographic IDs
unique_cells_df = chd_df[["row", "col"]].drop_duplicates().reset_index(drop=True)

# Map every distinct control point location to a unique time-series string identifier
chd_static_list = []
for idx, row in unique_cells_df.iterrows():
    r, c = int(row["row"]), int(row["col"])
    ts_name = f"cp_id_{idx}"
    chd_static_list.append(((0, r, c), ts_name))

# Bind this static template name dictionary to Stress Period 0 (MODFLOW carries it forward)
chd_spd = {0: chd_static_list}

# Assemble the actual transient continuous target values list
ts_records = []
for period_idx, group in chd_df.groupby("stress_period"):
    if period_idx >= nper_pump:
        continue # Strictly limit known transient inputs to the active window
    elapsed_time = float(period_idx * perlen)
    
    row_record = [elapsed_time]
    for idx, row in unique_cells_df.iterrows():
        r, c = int(row["row"]), int(row["col"])
        matching_head = group[(group["row"] == r) & (group["col"] == c)]["head"].values[0]
        row_record.append(float(matching_head))
        
    ts_records.append(tuple(row_record))

# EXTENDED TIMELINE FIX: Pad the remaining 18 weeks with stable tail-end conditions
last_group = chd_df[chd_df["stress_period"] == (nper_pump - 1)]

for current_p in range(nper_pump, nper_sim + 1):
    extended_elapsed_time = float(current_p * perlen)
    extended_row_record = [extended_elapsed_time]
    
    for idx, row in unique_cells_df.iterrows():
        r, c = int(row["row"]), int(row["col"])
        matching_head = last_group[(last_group["row"] == r) & (last_group["col"] == c)]["head"].values[0]
        extended_row_record.append(float(matching_head))
        
    ts_records.append(tuple(extended_row_record))

# Register these continuous series lists into a specialized FloPy time-series definition block
ts_names_record = [f"cp_id_{idx}" for idx in range(len(unique_cells_df))]
interpolation_methods = tuple(["linear"] * len(ts_names_record))

ts_dict = {
    "filename": f"{root}_donut_boundary_profiles.ts",
    "time_series_namerecord": ts_names_record,
    "timeseries": ts_records,
    "interpolation_methodrecord": [interpolation_methods]
}

# ==============================================================================
# 3. BUILD AND EXECUTE THE MODFLOW 6 SIMULATION
# ==============================================================================
print(f"Building continuous TS donut simulation in: {donut_ws}")
sim = flopy.mf6.MFSimulation(
    sim_name=f"{root}_donut_ts",
    exe_name=mf6_exe,
    sim_ws=donut_ws,
    verbosity_level=0,
)

flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper_sim, perioddata=tdis_rc)
flopy.mf6.ModflowIms(sim, complexity="MODERATE", outer_maximum=500, print_option="NONE")

gwf = flopy.mf6.ModflowGwf(sim, modelname=f"{root}_donut_ts", save_flows=True)

flopy.mf6.ModflowGwfdis(
    gwf, nlay=1, nrow=nrow, ncol=ncol, top=2477, botm=1000,
    delr=cell_dim, delc=cell_dim, xorigin=0.0, yorigin=0.0, idomain=idomain,
)

flopy.mf6.ModflowGwfnpf(gwf, k=0.005, icelltype=0)
iconvert = np.zeros((1, nrow, ncol), dtype=float)
flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, transient=True, iconvert=iconvert)

# Initialize initial heads background matrix
strt_array = np.zeros((1, nrow, ncol), dtype=float)
strt_gradient = np.linspace(2477.0, 1807.0, ncol)
for c in range(ncol):
    strt_array[0, :, c] = strt_gradient[c]

period_0_data = chd_df[chd_df["stress_period"] == 0]
for _, row in period_0_data.iterrows():
    strt_array[0, int(row["row"]), int(row["col"])] = row["head"]

flopy.mf6.ModflowGwfic(gwf, strt=strt_array)

# Re-apply regional General Head Boundaries (GHB) on active edges across all 70 periods
ghb_data = []
boundary_conductance = 0.05
for r in range(nrow):
    ghb_data.append(((0, r, 0), strt_gradient[0], boundary_conductance))      
    ghb_data.append(((0, r, ncol - 1), strt_gradient[-1], boundary_conductance)) 

ghb_period_data = {per: ghb_data for per in range(nper_sim)}
flopy.mf6.ModflowGwfghb(gwf, pname="ghb", stress_period_data=ghb_period_data)

# INSTANTIATE CONSTANT HEAD BOUNDARY CONSTRUCT CONTAINING INTUITIVE TIME SERIES INTERPOLATION
flopy.mf6.ModflowGwfchd(
    gwf,
    pname="chd",
    stress_period_data=chd_spd,
    timeseries=ts_dict
)

flopy.mf6.ModflowGwfoc(
    gwf, pname="oc",
    filename=f"{root}_donut_ts.oc",
    head_filerecord=f"{root}_donut_ts.hds",
    budget_filerecord=f"{root}_donut_ts.cbc",
    saverecord=[("head", "all"), ("budget", "all")],
    printrecord=[("head", "last"), ("budget", "last")]
)

print("Writing and running time-series donut model simulation...")
sim.write_simulation()
success, buff = sim.run_simulation(silent=False)

if success:
    print(f"\nTime-Series Donut model finished successfully! Outputs written to: {donut_ws}")
else:
    print("\nSimulation crashed. Check listing parameters for detail logs.")
