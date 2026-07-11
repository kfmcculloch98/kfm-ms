import os
import sys
import random
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import shutil

# ==============================================================================
#  PROJECT ROADMAP PATHS
# ==============================================================================
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parents[1]
sys.path.insert(0, str(project_root))

from codes.scripts.old.control_num_db import PEST_DIR

# ==============================================================================
# CONFIGURATION AND WORKSPACE SETUP
# ==============================================================================
VERBOSE = False

notebook_dir = os.getcwd()
MODFLOW_PATH = "../binaries/MODFLOW6/windows"
mf6_exe = os.path.join(notebook_dir, MODFLOW_PATH, "mf6.exe")

if not os.path.isfile(mf6_exe):
    flopy.utils.get_modflow(MODFLOW_PATH)

workspace_base = os.path.join("..", "sims")
workspace = None

# Simulation time discretisation
nper = 52
perlen = 7.0
nstp = 5
tsmult = 1.0

tdis_rc = [(perlen, nstp, tsmult) for _ in range(nper)]

# Spatial grid definition
cell_dim, ncol, nrow = 100, 92, 116
lx, ly = ncol * cell_dim, nrow * cell_dim


# ==============================================================================
# SMALL LOGGING HELPER
# ==============================================================================
def log(msg, level="info"):
    if level == "debug" and not VERBOSE:
        return
    prefix = {
        "info": "[INFO]",
        "debug": "[DEBUG]",
        "warn": "[WARN]",
        "error": "[ERROR]",
        "worker": "[WORKER]"
    }.get(level, "[INFO]")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} {prefix} {msg}")


# ==============================================================================
# HELPER: SAFE HEAD EXTRACTION
# ==============================================================================
def get_head_for_sp_by_search(hfile, sp):
    """
    Try to retrieve head data for a given stress period robustly:
      1) exact kper match
      2) expected last record in a 5-step stress period
      3) nearest earlier available record
    """
    keys = list(hfile.get_kstpkper())
    sp = int(sp)

    sp_keys = [k for k in keys if int(k[1]) == sp]
    if sp_keys:
        return hfile.get_data(kstpkper=sp_keys[-1])

    expected_idx = (sp + 1) * 5 - 1
    if 0 <= expected_idx < len(keys):
        return hfile.get_data(kstpkper=keys[expected_idx])

    earlier = [k for k in keys if int(k[1]) < sp]
    if earlier:
        return hfile.get_data(kstpkper=earlier[-1])

    raise ValueError(f"No head data found for stress period {sp}")


# ==============================================================================
# SIMULATION CORE BUILDER
# ==============================================================================
def setup_sim(name, wel_data, ws):
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

    ghb_period_data = {per: ghb_data for per in range(nper)}
    flopy.mf6.ModflowGwfghb(gwf, pname="ghb", stress_period_data=ghb_period_data)

    if wel_data:
        flopy.mf6.ModflowGwfwel(gwf, pname="wel", stress_period_data=wel_data)

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        filename=f"{name}.oc",
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord=[("head", "all"), ("budget", "last")],
        printrecord=[("head", "last"), ("budget", "last")]
    )

    return sim


# ==============================================================================
# PARALLEL SIReN WORKER (UNIT PULSE METHOD)
# ==============================================================================
def run_siren_well(task_info):
    idx, wr, wc, cp_cells, workspace, baseline_hds = task_info

    try:
        time.sleep(random.random() * 2)

        siren_ws = os.path.join(workspace, f"siren_{idx}")
        os.makedirs(siren_ws, exist_ok=True)
        name = f"siren_{idx}"

        scale_rate = 1000.0

        wel_spd = {
            p: [((0, wr, wc), -scale_rate)] if p == 0 else []
            for p in range(nper)
        }

        sim = setup_sim(name, wel_spd, siren_ws)
        sim.write_simulation()

        success, buff = sim.run_simulation(silent=True)
        if not success:
            raise RuntimeError(f"MF6 simulation failed for {name} at ({wr}, {wc})\n{buff}")

        hds_path = os.path.join(siren_ws, f"{name}.hds")
        if not os.path.exists(hds_path):
            raise FileNotFoundError(f"Missing head file: {hds_path}")

        h_file = flopy.utils.binaryfile.HeadFile(hds_path)

        n_cp = len(cp_cells)
        local_R = np.zeros((nper, n_cp))

        for p in range(nper):
            weekly_head_array = get_head_for_sp_by_search(h_file, p)
            for i, (r_cp, c_cp) in enumerate(cp_cells):
                pure_drawdown = baseline_hds[p, 0, r_cp, c_cp] - weekly_head_array[0, r_cp, c_cp]
                local_R[p, i] = pure_drawdown / scale_rate

        h_file.close()
        return idx, local_R

    except Exception as e:
        raise RuntimeError(f"Worker {idx} failed at basis location ({wr}, {wc}): {e}")


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
    SIMPLIFIED INTERNAL RESPONSE NETWORK
    """)

    root = input("Enter root name for this run: ").strip()
    workspace = os.path.abspath(os.path.join(workspace_base, root))

    if os.path.exists(workspace):
        log(f"Purging old simulation directory at: {workspace}")
        try:
            shutil.rmtree(workspace)
            time.sleep(1)
        except Exception as e:
            log(f"Could not automatically purge files: {e}", "warn")
            log("Please manually delete the folder or close background Python kernels.", "warn")

    margin = 25
    cp_buffer = 5

    log(f"Defined perimeter control points with a margin of {margin} cells...")
    log(f"Defined separation buffer between control points and wells of {cp_buffer} cells...")

    r_start, r_end = margin, (nrow - margin) - 1
    c_start, c_end = margin, (ncol - margin) - 1

    cp_cells = []
    for c in range(c_start, c_end + 1):
        cp_cells.extend([(r_start, c), (r_end, c)])
    for r in range(r_start + 1, r_end):
        cp_cells.extend([(r, c_start), (r, c_end)])

    n_cp = len(cp_cells)
    log(f"Defined {nrow} x {ncol} grid with {n_cp} perimeter control point cells...")

    # --------------------------------------------------------------------------
    # Baseline model
    # --------------------------------------------------------------------------
    log("Running no-pumping dynamic baseline simulation...")
    baseline_ws = os.path.join(workspace, "baseline")
    sim_b = setup_sim("baseline", {}, baseline_ws)
    sim_b.write_simulation()
    sim_b.run_simulation(silent=True)

    b_file = flopy.utils.binaryfile.HeadFile(os.path.join(baseline_ws, "baseline.hds"))
    baseline_hds = np.zeros((nper, 1, nrow, ncol))
    for p in range(nper):
        baseline_hds[p, :, :, :] = get_head_for_sp_by_search(b_file, p)
    b_file.close()

    rng = np.random.default_rng(seed=42)
    pumping_states = np.array([0, 1])

    # 20 real wells inside the interior zone
    inner_rows = range(r_start + cp_buffer, r_end - cp_buffer + 1)
    inner_cols = range(c_start + cp_buffer, c_end - cp_buffer + 1)
    interior_coords = [(r, c) for r in inner_rows for c in inner_cols]

    real_idx_choices = rng.choice(len(interior_coords), size=20, replace=False)
    real_locs_raw = [interior_coords[idx] for idx in real_idx_choices]

    log(f"Real well locations (row, col): {real_locs_raw}")

    real_wells_data = []
    for i, (r, c) in enumerate(real_locs_raw):
        base_rate = rng.uniform(273, 2730)
        binary_mask = rng.choice(pumping_states, size=nper)
        rates = base_rate * binary_mask
        real_wells_data.append({"well_id": i, "r": r, "c": c, "Q": rates})

    wel_spd_real = {
        p: [((0, w["r"], w["c"]), -w["Q"][p]) for w in real_wells_data]
        for p in range(nper)
    }

    rows = [
        {"well_id": w["well_id"], "r": w["r"], "c": w["c"], "sp": p, "q": w["Q"][p]}
        for w in real_wells_data
        for p in range(nper)
    ]
    master_truth_df = pd.DataFrame(rows)

    master_truth_path = os.path.join(workspace, "real", "master_truth.csv")
    os.makedirs(os.path.dirname(master_truth_path), exist_ok=True)
    master_truth_df.to_csv(master_truth_path, index=False)

    # --------------------------------------------------------------------------
    # SIReNs are colocated with the real wells
    # --------------------------------------------------------------------------
    candidate_coords = list(real_locs_raw)
    log(f"Defined {len(candidate_coords)} candidate basis locations colocated with the real wells...")

    # Save candidate metadata
    results_dir = current_script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    grid_candidates = []
    param_order_rows = []
    for sp in range(nper):
        for r, c in candidate_coords:
            grid_candidates.append({"r": r, "c": c, "sp": sp, "q": 0.0})
            param_order_rows.append({
                "parnme": f"q_r:{r}_c:{c}_sp:{sp}",
                "r": r,
                "c": c,
                "sp": sp
            })

    pd.DataFrame(grid_candidates).to_csv(results_dir / "grid_candidates.csv", index=False)
    pd.DataFrame(param_order_rows).to_csv(results_dir / "param_order.csv", index=False)
    real_ws = Path(master_truth_path).parent
    shutil.copy(results_dir / "grid_candidates.csv", real_ws / "grid_candidates.csv")
    shutil.copy(results_dir / "param_order.csv", real_ws / "param_order.csv")
    log(f"Saved grid_candidates.csv and param_order.csv to simulation directory.")

    # --------------------------------------------------------------------------
    # Run pumping truth simulation
    # --------------------------------------------------------------------------
    log("Running pumping simulation...")
    target_ws = os.path.join(workspace, "real")
    sim_t = setup_sim("real", wel_spd_real, target_ws)
    sim_t.write_simulation()
    success, buff = sim_t.run_simulation(silent=True)
    if not success:
        log("Real pumping simulation failed.", "error")
        print(buff)
        raise RuntimeError("Real pumping simulation failed.")

    hds_t_obj = flopy.utils.binaryfile.HeadFile(os.path.join(target_ws, "real.hds"))
    chd_export_records = []
    b_target_heads = []

    for r, c in cp_cells:
        for p in range(nper):
            absolute_head = get_head_for_sp_by_search(hds_t_obj, p)[0, r, c]
            b_target_heads.append(absolute_head)
            chd_export_records.append({
                "stress_period": p,
                "layer": 0,
                "row": r,
                "col": c,
                "head": absolute_head
            })
    b_target_heads = np.array(b_target_heads)
    hds_t_obj.close()

    b_file = flopy.utils.binaryfile.HeadFile(os.path.join(baseline_ws, "baseline.hds"))
    b_baseline_flat = []
    for r, c in cp_cells:
        for p in range(nper):
            b_baseline_flat.append(get_head_for_sp_by_search(b_file, p)[0, r, c])
    b_baseline_flat = np.array(b_baseline_flat)
    b_file.close()

    # --------------------------------------------------------------------------
    # Build response matrix
    # --------------------------------------------------------------------------
    n_siren = len(candidate_coords)
    log(f"Characterizing {n_siren} candidate basis locations in parallel...")

    tasks = [
        (j, int(candidate_coords[j][0]), int(candidate_coords[j][1]), cp_cells, workspace, baseline_hds)
        for j in range(n_siren)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_siren_well, tasks))

    R_full = np.zeros((nper, n_cp, n_siren))
    for idx, local_R in results:
        R_full[:, :, idx] = local_R

    # --------------------------------------------------------------------------
    # Construct G
    # --------------------------------------------------------------------------
    log("Assembling global surrogate matrix (G)...")
    G = np.zeros((nper * n_cp, nper * n_siren))

    p_obs, p_pump = np.indices((nper, nper))
    causal_mask = p_obs >= p_pump
    elapsed_weeks = p_obs - p_pump
    time_block_template = np.zeros((nper, nper))

    for i_cp in range(n_cp):
        row_start = i_cp * nper
        row_end = row_start + nper

        for j_siren in range(n_siren):
            col_start = j_siren * nper
            col_end = col_start + nper

            response_profile = R_full[:, i_cp, j_siren]
            time_block = time_block_template.copy()
            time_block[causal_mask] = response_profile[elapsed_weeks[causal_mask]]
            G[row_start:row_end, col_start:col_end] = time_block

    # --------------------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------------------
    log(f"Solving for {nper * n_siren} optimized pumping rates...")
    b_target_drawdown = b_baseline_flat - b_target_heads

    res = lsq_linear(G, b_target_drawdown, bounds=(0, np.inf), method="trf", max_iter=1000, verbose=0)
    log(f"Optimization complete. Solver iterations: {res.nit}")

    optimization_stats = {
        "root": root,
        "iterations": res.nit,
        "success": str(res.success),
        "status_message": res.message,
        "cost": res.cost
    }

    results_path = current_script_dir.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([optimization_stats]).to_csv(results_path / f"{root}_optimization_metadata.csv", index=False)

    q_flat_pest = res.x
    b_drawdown_match = G @ q_flat_pest
    b_head_match = b_baseline_flat - b_drawdown_match

    target_heads_matrix = b_target_heads.reshape(n_cp, nper).T
    match_heads_matrix = b_head_match.reshape(n_cp, nper).T

    cp_labels = [f"CP_r{r}_c{c}" for r, c in cp_cells]
    pd.DataFrame(target_heads_matrix, columns=cp_labels).rename_axis("SP").to_csv(
        results_path / f"{root}_target_absolute_heads.csv"
    )
    pd.DataFrame(match_heads_matrix, columns=cp_labels).rename_axis("SP").to_csv(
        results_path / f"{root}_surrogate_head_match.csv"
    )

    basis_labels = [f"Basis_r{r}_c{c}" for r, c in candidate_coords]
    q_opt = q_flat_pest.reshape(n_siren, nper).T
    pd.DataFrame(q_opt, columns=basis_labels).rename_axis("SP").to_csv(
        results_path / f"{root}_optimised_pumping.csv"
    )

    pd.DataFrame(chd_export_records).to_csv(
        results_path / f"{root}_chd_portable_boundaries.csv", index=False
    )

    np.save(results_path / "G_real_basis.npy", G)
    np.save((current_script_dir.parent / "pest") / "b_baseline_flat.npy", b_baseline_flat)

    log(f"Results successfully saved to: {results_path}")