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

import shutil

# ==============================================================================
# CONFIGURATION AND WORKSPACE SETUP
# ==============================================================================
notebook_dir = os.getcwd()
SCRIPT_DIR = Path(__file__).resolve().parent
PEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "pest"))

MODFLOW_PATH = "../binaries/MODFLOW6/windows"
mf6_exe = os.path.join(notebook_dir, MODFLOW_PATH, "mf6.exe")

if not os.path.isfile(mf6_exe):
    flopy.utils.get_modflow(MODFLOW_PATH)

workspace_base = os.path.join("..", "sims")
workspace = None

# Extended Time Horizon Configuration (Temporal Buffer Strategy)
nper_pump = 52      # Active pumping window we want to solve for
nper_sim = 70       # Total weeks simulated (52 active + 18 recovery tracking)
nper = nper_sim     # Point MODFLOW configurations to the full 70-week grid

perlen = 7.0        # Stress period length (days)
nstp = 5            # Internal solver steps per stress period
tsmult = 1.0        # Constant time step length multiplier

tdis_rc = [(perlen, nstp, tsmult) for _ in range(nper_sim)]

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
        nper=nper_sim,
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

    flopy.mf6.ModflowGwfnpf(gwf, k=0.005, icelltype=0)
    iconvert = np.zeros((1, nrow, ncol), dtype=float)
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, transient=True, iconvert=iconvert)

    strt_gradient = np.linspace(2477.0, 1807.0, ncol)
    strt_array = np.zeros((1, nrow, ncol), dtype=float)
    for c in range(ncol):
        strt_array[0, :, c] = strt_gradient[c]

    flopy.mf6.ModflowGwfic(gwf, strt=strt_array)

    ghb_data = []
    boundary_conductance = 0.05

    for r in range(nrow):
        ghb_data.append(((0, r, 0), strt_gradient[0], boundary_conductance))
        ghb_data.append(((0, r, ncol - 1), strt_gradient[-1], boundary_conductance))

    ghb_period_data = {per: ghb_data for per in range(nper_sim)}
    flopy.mf6.ModflowGwfghb(gwf, pname="ghb", stress_period_data=ghb_period_data)

    if wel_data:
        flopy.mf6.ModflowGwfwel(gwf, pname="wel", stress_period_data=wel_data)

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        filename=f"{name}.oc",
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord=[("head", "all"), ("budget", "all")],
        printrecord=[("head", "last"), ("budget", "last")]
    )

    return sim

# ==============================================================================
# PARALLEL SIReN WORKER (UNIT PULSE METHOD)
# ==============================================================================
def run_siren_well(task_info):
    """
    Simulates a 1-week Unit Pulse at a SIReN well across the extended time horizon.
    """
    idx, wr, wc, cp_cells, workspace, baseline_hds = task_info

    time.sleep(random.random() * 2)

    siren_ws = os.path.join(workspace, f"siren_{idx}")
    os.makedirs(siren_ws, exist_ok=True)
    name = f"siren_{idx}"

    scale_rate = 1000.0

    wel_spd = {}
    for p in range(nper_sim):
        if p == 0:
            wel_spd[p] = [((0, wr, wc), -scale_rate)]
        else:
            wel_spd[p] = []

    sim = setup_sim(name, wel_spd, siren_ws)
    sim.write_simulation()
    sim.run_simulation(silent=True)

    h_file = flopy.utils.binaryfile.HeadFile(os.path.join(siren_ws, f"{name}.hds"))

    n_cp = len(cp_cells)
    local_R = np.zeros((nper_sim, n_cp))

    for p in range(nper_sim):
        weekly_head_array = h_file.get_data(kstpkper=(4, p))
        for i, (r_cp, c_cp) in enumerate(cp_cells):
            pure_drawdown = baseline_hds[p, 0, r_cp, c_cp] - weekly_head_array[0, r_cp, c_cp]
            local_R[p, i] = pure_drawdown / scale_rate

    h_file.close()
    return idx, local_R

# ==============================================================================
# RUNTIME MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    print(r"""
    ███████╗██╗██████╗ ███████╗███╗   ██╗
    ██╔════╝██║██╔══██╗██╔════╝████╗  ██║
    ███████╗██║██████╔╝█████╗  ██╔██╗ ██║
    ╚════██║██║██╔══██╗██╔══╝  ██║╚██╗██║
    ███████║██║██║  ██║███████╗██║ ╚████║
    ╚══════╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝
    SIMPLIFIED INTERNAL RESPONSE NETWORK (RECTANGULAR EXTENSION)
    """)

    root = input("Enter root name for this run: ").strip()
    workspace = os.path.abspath(os.path.join(workspace_base, root))

    if os.path.exists(workspace):
        print(f"Purging old simulation directory at: {workspace}")
        try:
            shutil.rmtree(workspace)
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not automatically purge files: {e}")

    margin = 25
    well_buffer = 5

    r_start, r_end = margin, (nrow - margin) - 1
    c_start, c_end = margin, (ncol - margin) - 1
    cp_cells = []

    for c in range(c_start, c_end + 1):
        cp_cells.extend([(r_start, c), (r_end, c)])
    for r in range(r_start + 1, r_end):
        cp_cells.extend([(r, c_start), (r, c_end)])

    n_cp = len(cp_cells)

    n_real = 20
    inner_rows = range(r_start + well_buffer, r_end - well_buffer + 1)
    inner_cols = range(c_start + well_buffer, c_end - well_buffer + 1)
    inner_coords = [(r, c) for r in inner_rows for c in inner_cols]

    if len(inner_coords) < n_real:
        raise ValueError(f"Grid too small for {n_real} wells with a buffer of {well_buffer}.")

    spatial_rng = np.random.default_rng(seed=42)
    inner_coords_list = list(inner_coords)

    real_idx_choices = spatial_rng.choice(len(inner_coords_list), size=n_real, replace=False)
    real_locs_raw = [inner_coords_list[idx] for idx in real_idx_choices]

    siren_ratio = 1
    n_siren_target = math.floor(n_real * siren_ratio)
    sirens_per_well = n_siren_target / n_real
    directions = [
        (dr, dc)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    ]

    siren_locs_generated = []
    for i, (r, c) in enumerate(real_locs_raw):
        count_for_this_well = math.floor((i + 1) * sirens_per_well) - math.floor(i * sirens_per_well)
        dir_idx_choices = spatial_rng.choice(len(directions), size=count_for_this_well, replace=False)
        chosen_dirs = [directions[idx] for idx in dir_idx_choices]
        for dr, dc in chosen_dirs:
            siren_locs_generated.append((r + dr, c + dc))

    siren_locs = []
    for loc in real_locs_raw:
        if loc not in siren_locs:
            siren_locs.append(loc)

    n_siren = len(siren_locs)

    # --------------------------------------------------------------------------
    # STEP 1: RUN THE MANDATORY NO-PUMPING BASELINE SIMULATION (EXTENDED TO 70)
    # --------------------------------------------------------------------------
    print(f"Running no-pumping dynamic baseline simulation across {nper_sim} periods...")
    baseline_ws = os.path.join(workspace, "baseline")
    sim_b = setup_sim("baseline", {}, baseline_ws)
    sim_b.write_simulation()
    sim_b.run_simulation(silent=True)

    b_file = flopy.utils.binaryfile.HeadFile(os.path.join(baseline_ws, "baseline.hds"))
    baseline_hds = np.zeros((nper_sim, 1, nrow, ncol))
    for p in range(nper_sim):
        baseline_hds[p, :, :, :] = b_file.get_data(kstpkper=(4, p))
    b_file.close()

    # --------------------------------------------------------------------------
    # STEP 2: GENERATE AND RUN SYNTHETIC TRUTH MODEL (CLIP ACTIVE PUMPING AT 52)
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    real_wells_data = []
    pumping_states = np.array([0, 1])

    for i, (r, c) in enumerate(real_locs_raw):
        base_rate = rng.uniform(273, 2730)
        binary_mask = rng.choice(pumping_states, size=nper_pump)  # Active generation limited to 52
        rates_active = base_rate * binary_mask

        # Append 18 weeks of zero-pumping tail recovery
        rates_extended = np.concatenate([rates_active, np.zeros(nper_sim - nper_pump)])

        real_wells_data.append({
            "well_id": i, "r": r, "c": c, "Q": rates_extended
        })

    wel_spd_real = {
        p: [((0, w["r"], w["c"]), -w["Q"][p]) for w in real_wells_data]
        for p in range(nper_sim)
    }

    # Explicitly track and export history for only the 52 active verification weeks
    rows = [
        {"well_id": w["well_id"], "r": w["r"], "c": w["c"], "sp": p, "q": w["Q"][p]}
        for w in real_wells_data
        for p in range(nper_pump)
    ]
    master_truth_df = pd.DataFrame(rows)

    master_truth_save_path = os.path.join(workspace, "real", "master_truth.csv")
    os.makedirs(os.path.dirname(master_truth_save_path), exist_ok=True)
    master_truth_df.to_csv(master_truth_save_path, index=False)

    print(f"Running pumping truth simulation through full {nper_sim}-week window...")
    target_ws = os.path.join(workspace, "real")
    sim_t = setup_sim("real", wel_spd_real, target_ws)
    sim_t.write_simulation()
    sim_t.run_simulation(silent=True)

    hds_t_obj = flopy.utils.binaryfile.HeadFile(os.path.join(target_ws, "real.hds"))
    chd_export_records = []
    b_target_heads = []

    # Map out the full target data across all 70 tracking weeks
    for r, c in cp_cells:
        for p in range(nper_sim):
            absolute_head = hds_t_obj.get_data(kstpkper=(4, p))[0, r, c]
            b_target_heads.append(absolute_head)
            if p < nper_pump:
                # Export portable baseline references for evaluation window
                chd_export_records.append({
                    "stress_period": p,
                    "layer": 0,
                    "row": r,
                    "col": c,
                    "head": absolute_head
                })

    b_target_heads = np.array(b_target_heads)
    hds_t_obj.close()

    # --------------------------------------------------------------------------
    # STEP 3: PARALLEL RESPONSE MATRIX GENERATION (EXTENDED WINDOW RESPONSE)
    # --------------------------------------------------------------------------
    print(f"Characterizing {n_siren} SIReN response columns across {nper_sim} steps...")
    tasks = [
        (j, siren_locs[j][0], siren_locs[j][1], cp_cells, workspace, baseline_hds)
        for j in range(n_siren)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_siren_well, tasks))

    R_full = np.zeros((nper_sim, n_cp, n_siren))
    for idx, local_R in results:
        R_full[:, :, idx] = local_R

    # --------------------------------------------------------------------------
    # STEP 4: CONSTRUCT RECTANGULAR CONVOLUTION MATRICES (G)
    # --------------------------------------------------------------------------
    print("Assembling rectangular surrogate matrix (G) via block convolution allocation...")

    # Rows: 70 Observation steps per channel | Columns: 52 Optimized pumping slots
    G = np.zeros((nper_sim * n_cp, nper_pump * n_siren))

    for i_cp in range(n_cp):
        row_start = i_cp * nper_sim
        row_end = row_start + nper_sim
        for j_siren in range(n_siren):
            col_start = j_siren * nper_pump
            col_end = col_start + nper_pump
            response_profile = R_full[:, i_cp, j_siren]  # Length 70 response timeline
            time_block = np.zeros((nper_sim, nper_pump))
            for p_pump in range(nper_pump):
                # Slid down response index over time to preserve causality bounds
                time_block[p_pump:, p_pump] = response_profile[:nper_sim - p_pump]
            G[row_start:row_end, col_start:col_end] = time_block

    # --------------------------------------------------------------------------
    # STEP 5: OPTIMIZATION SOLVER & METRICS EXPORT
    # --------------------------------------------------------------------------
    print(f"Solving bounded system for {nper_pump * n_siren} active pumping targets...")

    b_baseline_flat = np.array([baseline_hds[p, 0, r, c] for r, c in cp_cells for p in range(nper_sim)])
    b_target_drawdown = b_baseline_flat - b_target_heads

    # G has (70 * n_cp) rows and (52 * n_siren) columns. Returns 52 rates per well.
    res = lsq_linear(
        G,
        b_target_drawdown,
        bounds=(0, np.inf),
        method="trf",
        max_iter=1000,
        verbose=0
    )

    print(f"Optimization complete. Solver iterations: {res.nit}")

    optimization_stats = {
        "root": root,
        "iterations": res.nit,
        "success": str(res.success),
        "status_message": res.message,
        "cost": res.cost
    }

    script_dir = Path(__file__).resolve().parent
    results_path = script_dir.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    stats_df = pd.DataFrame([optimization_stats])
    stats_df.to_csv(results_path / f"{root}_optimization_metadata.csv", index=False)

    q_flat_pest = res.x
    b_drawdown_match_full = G @ q_flat_pest
    b_head_match_full = b_baseline_flat - b_drawdown_match_full

    # Pivot array blocks back into Standard Report Output Structure (Restricted to 52 verification weeks)
    q_opt = q_flat_pest.reshape((n_siren, nper_pump)).T

    # Reshape total head matrices to isolate only the active 52-week evaluations for your report plots
    target_heads_matrix = b_target_heads.reshape((n_cp, nper_sim))[:, :nper_pump].T
    match_heads_matrix = b_head_match_full.reshape((n_cp, nper_sim))[:, :nper_pump].T

    chd_df = pd.DataFrame(chd_export_records)
    chd_df.to_csv(results_path / f"{root}_chd_portable_boundaries.csv", index=False)
    pd.DataFrame(target_heads_matrix).to_csv(results_path / f"{root}_target_absolute_heads.csv", index_label="SP")
    pd.DataFrame(match_heads_matrix).to_csv(results_path / f"{root}_surrogate_head_match.csv", index_label="SP")
    pd.DataFrame(q_opt).to_csv(results_path / f"{root}_optimised_pumping.csv", index_label="SP")

    # Save base structural matrix files
    np.save(results_path / "G_real_basis.npy", G)
    np.save(os.path.join(PEST_DIR, "b_baseline_flat.npy"), b_baseline_flat)

    print(f"Rectangular matrix stabilization complete. Outputs written to: {results_path}")