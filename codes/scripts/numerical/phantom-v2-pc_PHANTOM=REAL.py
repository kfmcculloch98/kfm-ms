import os
import random
import time
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# ==============================================================================
# CONFIGURATION AND WORKSPACE SETUP
# ==============================================================================
notebook_dir = os.getcwd()

MODFLOW_PATH = "../binaries/MODFLOW6/windows"
mf6_exe = os.path.join(notebook_dir, MODFLOW_PATH, "mf6.exe")

# Verify or download the MODFLOW 6 executable locally via flopy
if not os.path.isfile(mf6_exe):
    flopy.utils.get_modflow(MODFLOW_PATH)

workspace_base = os.path.join("..", "sims")
workspace = None

# Simulation time discretisation
nper = 52       # Number of stress periods (weeks)
perlen = 7.0    # Stress period length (days)
nstp = 5        # Internal solver steps per stress period
tsmult = 1.0    # Constant time step length multiplier

# Structural time-discretisation list across all 52 periods
tdis_rc = [(perlen, nstp, tsmult) for _ in range(nper)]

# Spatial grid definition
cell_dim, ncol, nrow = 100, 92, 116 
lx, ly = ncol * cell_dim, nrow * cell_dim

# ==============================================================================
# SIMULATION CORE BUILDER
# ==============================================================================
def setup_sim(name, wel_data, ws):
    """
    Creates a standardized MODFLOW 6 groundwater flow simulation.
    """
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=mf6_exe,
        sim_ws=ws,
        verbosity_level=0,
    )
    
    flopy.mf6.ModflowTdis(
        sim, 
        time_units="DAYS", 
        nper=nper, 
        perioddata=tdis_rc
    )

    flopy.mf6.ModflowIms(
        sim,
        complexity="MODERATE",
        outer_maximum=500,
        print_option="NONE",
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    idomain = np.ones((1, nrow, ncol), dtype=int)
    
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        top=2477,
        botm=1000,
        delr=cell_dim,
        delc=cell_dim,
        xorigin=0.0,
        yorigin=0.0,
        idomain=idomain,
    )

    # Bedrock aquifer hydraulic parameters
    flopy.mf6.ModflowGwfnpf(gwf, k=0.005, icelltype=0) # Hydraulic conductivity (m/day) and cell type (confined)
    iconvert = np.zeros((1, nrow, ncol), dtype=float) # Confined behavior
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, transient=True, iconvert=iconvert) # Specific storage (1/m) and transient storage behavior

    # Linear left-to-right regional gradient
    strt_gradient = np.linspace(2477.0, 1807.0, ncol)
    strt_array = np.zeros((1, nrow, ncol), dtype=float)
    for c in range(ncol):
        strt_array[0, :, c] = strt_gradient[c]

    flopy.mf6.ModflowGwfic(gwf, strt=strt_array)

    # Regional boundaries to manage boundary-driven dynamic transient trends
    ghb_data = []
    boundary_conductance = 0.05  # m2/day

    for r in range(nrow):
        ghb_data.append(((0, r, 0), strt_gradient[0], boundary_conductance))      # Left source
        ghb_data.append(((0, r, ncol - 1), strt_gradient[-1], boundary_conductance)) # Right sink

    ghb_period_data = {per: ghb_data for per in range(nper)}
    flopy.mf6.ModflowGwfghb(gwf, pname="ghb", stress_period_data=ghb_period_data)

    # Apply conditional well stress package
    if wel_data:
        flopy.mf6.ModflowGwfwel(gwf, pname="wel", stress_period_data=wel_data)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        saverecord=[("head", "last")], # Save only the final step of each week
    )

    return sim

# ==============================================================================
# PARALLEL PHANTOM WORKER (UNIT PULSE METHOD)
# ==============================================================================
def run_phantom_well(task_info):
    """
    Simulates a 1-week Unit Pulse at a phantom well and extracts pure drawdown 
    by subtracting from the natural dynamic baseline heads.
    """
    idx, wr, wc, cp_cells, workspace, baseline_hds = task_info

    # Stagger execution initialization to stabilize parallel File I/O
    time.sleep(random.random() * 2)

    phantom_ws = os.path.join(workspace, f"phantom_{idx}")
    os.makedirs(phantom_ws, exist_ok=True)
    name = f"phantom_{idx}"

    scale_rate = 1000.0

    # Unit Pulse Scheme: Pumping active ONLY during stress period 0
    wel_spd = {}
    for p in range(nper):
        if p == 0:
            wel_spd[p] = [((0, wr, wc), -scale_rate)]
        else:
            wel_spd[p] = [] # Empty list keeps well off for recovery periods

    sim = setup_sim(name, wel_spd, phantom_ws)
    sim.write_simulation()
    sim.run_simulation(silent=True)

    h_file = flopy.utils.binaryfile.HeadFile(os.path.join(phantom_ws, f"{name}.hds"))
    h_data = h_file.get_alldata()
    h_file.close()

    n_cp = len(cp_cells)
    local_R = np.zeros((nper, n_cp))
    
    # Calculate pure drawdown relative to the dynamic baseline model
    for p in range(nper):
        for i, (r_cp, c_cp) in enumerate(cp_cells):
            pure_drawdown = baseline_hds[p, 0, r_cp, c_cp] - h_data[p, 0, r_cp, c_cp]
            local_R[p, i] = pure_drawdown / scale_rate

    return idx, local_R

# ==============================================================================
# RUNTIME MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    root = input("Enter root name for this run: ").strip()
    workspace = os.path.abspath(os.path.join(workspace_base, root))

    # Spatial domain filtering and compliance boundaries
    margin = 25 
    well_buffer = 5 

    print(f"Defined perimeter control points with a margin of {margin} cells...")
    print(f"Defined separation buffer between control points and wells of {well_buffer} cells...")

    r_start, r_end = margin, (nrow - margin) - 1
    c_start, c_end = margin, (ncol - margin) - 1
    cp_cells = []

    for c in range(c_start, c_end + 1):
        cp_cells.extend([(r_start, c), (r_end, c)])
    for r in range(r_start + 1, r_end):
        cp_cells.extend([(r, c_start), (r, c_end)])

    n_cp = len(cp_cells)
    print(f"Defined {nrow} x {ncol} grid with {n_cp} perimeter cells...")

    # Establish well allocation zones
    n_real = 20
    inner_rows = range(r_start + well_buffer, r_end - well_buffer + 1)
    inner_cols = range(c_start + well_buffer, c_end - well_buffer + 1)
    inner_coords = [(r, c) for r in inner_rows for c in inner_cols]

    if len(inner_coords) < n_real:
        raise ValueError(f"Grid too small for {n_real} wells with a buffer of {well_buffer}.")

    random.seed(42)
    real_locs_raw = random.sample(inner_coords, n_real)
    print(f"Raw sampled well locations (row, col): {real_locs_raw}")

    # Process and map localized spatial phantom layouts
    phantom_ratio = 1
    n_phantom_target = math.floor(n_real * phantom_ratio)
    phantoms_per_well = n_phantom_target / n_real
    directions = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]

    phantom_locs_generated = []
    for i, (r, c) in enumerate(real_locs_raw):
        count_for_this_well = math.floor((i + 1) * phantoms_per_well) - math.floor(i * phantoms_per_well)
        chosen_dirs = random.sample(directions, count_for_this_well)
        for dr, dc in chosen_dirs:
            phantom_locs_generated.append((r + dr, c + dc))

    # Keep alignment simple for basis processing (drop duplicates/overlaps)
    phantom_locs = []
    for loc in real_locs_raw:
        if loc not in phantom_locs:
            phantom_locs.append(loc)
            
    n_phantom = len(phantom_locs)
    print(f"Set up {n_phantom} unique PHANTOM wells for basis testing...")

    # --------------------------------------------------------------------------
    # STEP 1: RUN THE MANDATORY NO-PUMPING BASELINE SIMULATION
    # --------------------------------------------------------------------------
    print("Running no-pumping dynamic baseline simulation...")
    baseline_ws = os.path.join(workspace, "baseline")
    sim_b = setup_sim("baseline", {}, baseline_ws)
    sim_b.write_simulation()
    sim_b.run_simulation(silent=True)

    b_file = flopy.utils.binaryfile.HeadFile(os.path.join(baseline_ws, "baseline.hds"))
    baseline_hds = b_file.get_alldata() # shape: (nper, nlay, nrow, ncol)
    b_file.close()

    # --------------------------------------------------------------------------
    # STEP 2: GENERATE AND RUN SYNTHETIC TRUTH MODEL
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    real_wells_data = []
    pumping_states = np.array([0, 1])
    
    for i, (r, c) in enumerate(phantom_locs):
        base_rate = rng.uniform(273, 2730)
        binary_mask = rng.choice(pumping_states, size=nper)
        rates = base_rate * binary_mask
        
        real_wells_data.append({
            "well_id": i, "r": r, "c": c, "Q": rates
        })

    wel_spd_real = {p: [((0, w["r"], w["c"]), -w["Q"][p]) for w in real_wells_data] for p in range(nper)}

    # Save tracking history
    rows = [{"well_id": w["well_id"], "r": w["r"], "c": w["c"], "sp": p, "q": w["Q"][p]} for w in real_wells_data for p in range(nper)]
    master_truth_df = pd.DataFrame(rows)
    master_truth_save_path = os.path.join(workspace, "real", "master_truth.csv")
    os.makedirs(os.path.dirname(master_truth_save_path), exist_ok=True)
    master_truth_df.to_csv(master_truth_save_path, index=False)

    print("Running reference truth simulation...")
    target_ws = os.path.join(workspace, "real")
    sim_t = setup_sim("real", wel_spd_real, target_ws)
    sim_t.write_simulation()
    sim_t.run_simulation(silent=True)

    hds_t_obj = flopy.utils.binaryfile.HeadFile(os.path.join(target_ws, "real.hds"))
    hds_t = hds_t_obj.get_alldata()
    hds_t_obj.close()

    # Calculate absolute drawdown vector isolated from dynamic regional shifts
    b_target = []
    for r, c in cp_cells:
        for p in range(nper):
            pure_target_dd = baseline_hds[p, 0, r, c] - hds_t[p, 0, r, c]
            b_target.append(pure_target_dd)
    b_target = np.array(b_target)

    # --------------------------------------------------------------------------
    # STEP 3: PARALLEL RESPONSE MATRIX GENERATION (SUPERPOSITION SURROGATE)
    # --------------------------------------------------------------------------
    print(f"Characterising {n_phantom} PHANTOM wells in parallel...")
    tasks = [
        (j, phantom_locs[j][0], phantom_locs[j][1], cp_cells, workspace, baseline_hds)
        for j in range(n_phantom)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_phantom_well, tasks))

    R_full = np.zeros((nper, n_cp, n_phantom))
    for idx, local_R in results:
        R_full[:, :, idx] = local_R

    # --------------------------------------------------------------------------
    # STEP 4: CONSTRUCT CONVOLUTION MATRICES (G)
    # --------------------------------------------------------------------------
    print("Assembling global surrogate matrix (G) using block convolution allocation...")
    G = np.zeros((nper * n_cp, nper * n_phantom))
    
    p_obs, p_pump = np.indices((nper, nper))
    causal_mask = p_obs >= p_pump
    elapsed_weeks = p_obs - p_pump
    time_block_template = np.zeros((nper, nper))

    for i_cp in range(n_cp):
        row_start = i_cp * nper
        row_end = row_start + nper
        
        for j_phan in range(n_phantom):
            col_start = j_phan * nper
            col_end = col_start + nper
            
            response_profile = R_full[:, i_cp, j_phan]
            
            time_block = time_block_template.copy()
            # Maps pulse response + transient recovery through time
            time_block[causal_mask] = response_profile[elapsed_weeks[causal_mask]]
            
            G[row_start:row_end, col_start:col_end] = time_block

    # --------------------------------------------------------------------------
    # STEP 5: OPTIMIZATION SOLVER & METRICS EXPORT
    # --------------------------------------------------------------------------
    print(f"Solving for {nper * n_phantom} optimized pumping rates...")
    res = lsq_linear(G, b_target, bounds=(0, np.inf), method="trf", max_iter=1000, verbose=0)

    print(f"Optimization complete. Solver iterations: {res.nit}")

    # Build a metadata dictionary of the solver results
    optimization_stats = {
        "root": root,
        "iterations": res.nit,
        "success": str(res.success),
        "status_message": res.message,
        "cost": res.cost
    }

    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    # Save to a diagnostic CSV in the results folder
    stats_df = pd.DataFrame([optimization_stats])
    stats_df.to_csv(results_path / f"{root}_optimization_metadata.csv", index=False)
    print(f"Optimization metadata successfully saved to: {results_path}")

    # Save the optimized pumping rates and corresponding drawdown matches
    q_flat_pest = res.x
    b_match = G @ q_flat_pest

    # Structural re-indexing to spreadsheet format (Rows = Time, Columns = Location)
    q_opt = q_flat_pest.reshape((n_phantom, nper)).T
    target_reshaped = b_target.reshape((n_cp, nper)).T
    match_reshaped = b_match.reshape((n_cp, nper)).T

    # Save the results to CSV files for further analysis
    pd.DataFrame(target_reshaped).to_csv(results_path / f"{root}_target_drawdown.csv", index_label="SP")
    pd.DataFrame(q_opt).to_csv(results_path / f"{root}_optimised_pumping.csv", index_label="SP")
    pd.DataFrame(match_reshaped).to_csv(results_path / f"{root}_surrogate_match.csv", index_label="SP")
    np.save(results_path / "G_real_basis.npy", G)

    print(f"Results successfully saved to: {results_path}")