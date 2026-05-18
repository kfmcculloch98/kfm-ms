#######################
### import packages ###
#######################

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

#######################################
### configure MODFLOW and workspace ###
#######################################

notebook_dir = os.getcwd()

MODFLOW_PATH = "../binaries/MODFLOW6/windows"
mf6_exe = os.path.join(notebook_dir, MODFLOW_PATH, "mf6.exe")

# check whether the modflow executable is available locally
# if not, attempt to download it using flopy's utility function
if not os.path.isfile(mf6_exe):
    flopy.utils.get_modflow(MODFLOW_PATH)

# set the base workspace for all simulations
workspace_base = os.path.join("..", "sims")
workspace = None

########################
### simulation setup ###
########################

# stress period length (days) and number of periods
dt, nper = 7.0, 10

# grid parameters: cell size and number of rows and columns
cell_dim, ncol, nrow = 1.0, 50, 50
lx, ly = ncol * cell_dim, nrow * cell_dim

def setup_sim(name, wel_data, ws):
    """
    create a MODFLOW 6 simulation.

    parameters:
    - name: simulation name used for file naming
    - wel_data: well stress period data
    - ws: workspace directory where simulation files will be written
    """

    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=mf6_exe,
        sim_ws=ws,
        verbosity_level=0,
    )

    # define temporal discretisation for the full simulation
    flopy.mf6.ModflowTdis(
        sim,
        nper=nper,
        perioddata=[(dt, 1, 1.0)] * nper,
    )

    # set up the iterative model solution package
    flopy.mf6.ModflowIms(
        sim,
        complexity="MODERATE",
        outer_maximum=500,
        print_option="NONE",
    )

    # create the groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        top=0.0,
        botm=[-100.0],
        delr=cell_dim,
        delc=cell_dim,
        xorigin=0.0,
        yorigin=0.0,
    )

    # assign hydraulic and storage properties
    flopy.mf6.ModflowGwfnpf(gwf, k=1e-3, icelltype=0)
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, transient=True)

    # set initial heads
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)

    flopy.mf6.ModflowGwfwel(gwf, pname="wel", stress_period_data=wel_data)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        saverecord=[("HEAD", "ALL")],
    )

    return sim

#################################
### parallel processing setup ###
#################################

def run_phantom_well(task_info):
    """
    worker function used to characterise the influence of a single phantom well.

    parameters:
    - task_info: tuple containing (idx, wr, wc, cp_cells, workspace)
    """

    idx, wr, wc, cp_cells, workspace = task_info

    # stagger process start times to reduce file i/o conflicts
    time.sleep(random.random() * 2)

    # create a unique workspace for each phantom well
    phantom_ws = os.path.join(workspace, f"phantom_{idx}")
    os.makedirs(phantom_ws, exist_ok=True)

    name = f"phantom_{idx}"

    # apply a unit pumping rate at the phantom well for all stress periods
    wel_spd = {p: [((0, wr, wc), -1.0)] for p in range(nper)}

    sim = setup_sim(name, wel_spd, phantom_ws)
    sim.write_simulation()
    sim.run_simulation(silent=True)

    # extract simulated heads at the perimeter control points
    h_file = flopy.utils.binaryfile.HeadFile(os.path.join(phantom_ws, f"{name}.hds"))
    h_data = h_file.get_alldata()

    # compute the local response vector for this phantom well
    n_cp = len(cp_cells)
    local_R = np.zeros((nper, n_cp))
    for p in range(nper):
        for i, (r_cp, c_cp) in enumerate(cp_cells):
            local_R[p, i] = 0.0 - h_data[p, 0, r_cp, c_cp]

    h_file.close()
    return idx, local_R

######################
### code execution ###
######################

if __name__ == "__main__":

    root = input("Enter root name for this run: ").strip()
    workspace = os.path.abspath(os.path.join(workspace_base, root))

    # define perimeter control points with a fixed margin from the domain boundary
    margin = 5

    r_start, r_end = margin, (nrow - margin) - 1
    c_start, c_end = margin, (ncol - margin) - 1
    cp_cells = []

    # top and bottom perimeter cells
    for c in range(c_start, c_end + 1):
        cp_cells.extend([(r_start, c), (r_end, c)])

    # left and right perimeter cells, excluding corners
    for r in range(r_start + 1, r_end):
        cp_cells.extend([(r, c_start), (r, c_end)])

    n_cp = len(cp_cells)
    print(f"Defined {nrow} x {ncol} grid with {n_cp} perimeter cells...")

    # define phantom wells at the same locations as the perimeter cells
    phantom_locs = cp_cells
    n_phantom = len(phantom_locs)
    print(f"Set up {n_phantom} PHANTOM wells along the perimeter...")

    # define the number of real wells to simulate
    n_real = 10
    inner_rows = range(r_start + 1, r_end)
    inner_cols = range(c_start + 1, c_end)
    inner_coords = [(r, c) for r in inner_rows for c in inner_cols]

    # select random locations for the real wells from the inner grid
    random.seed(2026)
    real_locs = random.sample(inner_coords, n_real)
    print(f"Real well locations (row, col): {real_locs}")

    print(f"Running simulation for {n_real} real wells...")
    real_wells_data = []
    for r, c in real_locs:
        real_wells_data.append(
            {
                "r": r,
                "c": c,
                # assign random pumping rates within specified bounds
                "Q": np.random.uniform(0.005, 0.02, nper),
            }
        )

    # assemble stress period data for the real wells
    wel_spd_real = {
        p: [((0, w["r"], w["c"]), -w["Q"][p]) for w in real_wells_data]
        for p in range(nper)
    }

    # store the real well schedule for PEST
    rows = []
    for i, w in enumerate(real_wells_data):
        for p in range(nper):
            rows.append(
                {
                    "well_id": i,
                    "r": w["r"],
                    "c": w["c"],
                    "sp": p,
                    "q": w["Q"][p],
                }
            )

    master_truth_df = pd.DataFrame(rows)
    master_truth_save_path = os.path.join(workspace, "real", "master_truth.csv")
    os.makedirs(os.path.dirname(master_truth_save_path), exist_ok=True)
    master_truth_df.to_csv(master_truth_save_path, index=False)

    # run the reference simulation to generate the target drawdown
    target_ws = os.path.join(workspace, "real")
    sim_t = setup_sim("real", wel_spd_real, target_ws)
    sim_t.write_simulation()
    sim_t.run_simulation(silent=True)

    # extract target drawdown at the perimeter control points
    hds_t_obj = flopy.utils.binaryfile.HeadFile(os.path.join(target_ws, "real.hds"))
    hds_t = hds_t_obj.get_alldata()
    hds_t_obj.close()

    b_target = np.array(
        [0.0 - hds_t[p, 0, r, c] for p in range(nper) for r, c in cp_cells]
    )

    # characterise the response of each phantom well in parallel
    print(f"characterising {n_phantom} phantom wells in parallel...")
    tasks = [
        (j, phantom_locs[j][0], phantom_locs[j][1], cp_cells, workspace)
        for j in range(n_phantom)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_phantom_well, tasks))

    # assemble the local response vectors into a global response tensor
    R_full = np.zeros((nper, n_cp, n_phantom))
    for idx, local_R in results:
        R_full[:, :, idx] = local_R

    ####################
    ### optimisation ###
    ####################

    # build the global surrogate matrix
    print("Assembling global surrogate matrix (G)...")
    R_shifted = np.zeros_like(R_full)
    R_shifted[1:, :, :] = R_full[:-1, :, :]
    U_inc_all = R_full - R_shifted

    # place the incremental response blocks into the global matrix
    G = np.zeros((nper * n_cp, nper * n_phantom))
    for pp in range(nper):
        block = U_inc_all[:nper - pp, :, :].reshape(-1, n_phantom)
        G[pp * n_cp:, pp * n_phantom:(pp + 1) * n_phantom] = block

    # apply l2 regularisation to stabilise the inverse problem
    alpha = 0.01
    G_reg = np.vstack([G, np.eye(nper * n_phantom) * alpha])
    b_reg = np.concatenate([b_target, np.zeros(nper * n_phantom)])

    # solve for the optimised pumping rates
    print(f"Solving for {nper * n_phantom} optimised pumping rates... this may take a while...")
    res = lsq_linear(G_reg, b_reg, bounds=(0, 500))
    q_opt = res.x.reshape((nper, n_phantom))
    print("Success! Optimisation complete. Calculating the surrogate match and saving results...")

    # calculate the surrogate-predicted drawdown at the perimeter control points
    b_match = G @ q_opt.flatten()

    #######################
    ### process results ###
    #######################

    # reshape results for comparison and visualisation
    match_reshaped = b_match.reshape((nper, n_cp))
    target_reshaped = b_target.reshape((nper, n_cp))

    # define the results directory
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    # save model outputs
    pd.DataFrame(target_reshaped).to_csv(
        results_path / f"{root}_target_drawdown.csv",
        index_label="SP",
    )
    pd.DataFrame(q_opt).to_csv(
        results_path / f"{root}_optimised_pumping.csv",
        index_label="SP",
    )
    pd.DataFrame(match_reshaped).to_csv(
        results_path / f"{root}_surrogate_match.csv",
        index_label="SP",
    )
    np.save("../pest/surrogate_G.npy", G)

    print(f"Results saved to: {results_path}")