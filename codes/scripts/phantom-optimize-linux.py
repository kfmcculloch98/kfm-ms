#######################
### import packages ###
#######################

import os
import shutil
import time
import numpy as np
import flopy
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import random
from pathlib import Path

#######################################
### configure MODFLOW and workspace ###
#######################################

notebook_dir = os.getcwd()

MODFLOW_PATH = "../binaries/MODFLOW6/linux"
mf6_elf = os.path.join(notebook_dir, MODFLOW_PATH, "mf6.elf")

# check if MODFLOW is accessible at the specified path
# if not, attempt to download it using flopy's utility function
if not os.path.isfile(mf6_elf):
    flopy.utils.get_modflow(MODFLOW_PATH)

# set root name and workspace location
workspace_base = os.path.join("..", "sims")
workspace = None

########################
### simulation setup ###
########################

# stress period length (days) and number of periods
dt, nper = 7.0, 10

# grid parameters: cell size and number of rows/columns
cell_dim, ncol, nrow = 2.0, 50, 50
lx, ly = ncol * cell_dim, nrow * cell_dim

def setup_sim(name, wel_data, ws):
    """
    Creates a MODFLOW 6 simulation.
    Parameters:
    - name: Unique name for the simulation (used for file naming)
    - wel_data: Dictionary of well stress period data, e.g. {0: [((0, r, c), Q), ...], 1: [...], ...}
    - ws: Workspace directory where simulation files will be written
    """

    # create simulation
    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name=mf6_elf, sim_ws=ws, verbosity_level=0)

    # create time discretization: 
    # nper stress period of length 'dt' days, with 1 time step and a multiplier of 1.0
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=[(dt, 1, 1.0)]*nper)

    # create iterative model solution and register the groundwater flow model with it
    flopy.mf6.ModflowIms(sim, complexity="MODERATE", outer_maximum=500, print_option="NONE")
    
    # create groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol, top=0.0, botm=[-10.0],
                            delr=cell_dim, delc=cell_dim, xorigin=-lx/2, yorigin=-ly/2)
    
    # hydraulic properties: horizontal conductivity (k) and cell type (icelltype=0 for confined)
    flopy.mf6.ModflowGwfnpf(gwf, k=1e-3, icelltype=0) 

    # storage properties: specific storage (ss) and transient behavior
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, transient=True)

    # initial conditions: starting head (strt) of 0.0 everywhere
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)

    flopy.mf6.ModflowGwfwel(gwf, pname='wel', stress_period_data=wel_data)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", saverecord=[("HEAD", "ALL")])

    return sim

#################################
### parallel processing setup ###
#################################

def run_phantom_well(task_info):
    """
    Worker function to characterize the influence of a single Phantom Well.
    Parameters:
    - task_info: Tuple containing (idx, wr, wc, cp_cells) where:
    - idx: Index of the phantom well (for naming)
    - wr, wc: Row and column of the phantom well
    - cp_cells: List of perimeter cell coordinates to extract drawdown from
    """

    idx, wr, wc, cp_cells, workspace = task_info

    # add a random sleep to stagger the start times of the parallel simulations
    # reduces file I/O conflicts
    time.sleep(random.random() * 2) 
    
    # create a unique workspace for this phantom to avoid file conflicts
    phantom_ws = os.path.join(workspace, f"phantom_{idx}")
    os.makedirs(phantom_ws, exist_ok=True)
    
    name = f"phantom_{idx}"

    # unit pumping rate of -1.0 (extraction) at the phantom well for all stress periods
    # characterizes the "fingerpint" of this phantom
    wel_spd = {p: [((0, wr, wc), -1.0)] for p in range(nper)}
    
    sim = setup_sim(name, wel_spd, phantom_ws)
    sim.write_simulation()
    sim.run_simulation(silent=True)
    
    # extract head data at perimeter cells for all stress periods
    h_file = flopy.utils.binaryfile.HeadFile(os.path.join(phantom_ws, f"{name}.hds"))
    h_data = h_file.get_alldata()
    
    # compute local response vector for this phantom well: 
    # drawdown at each perimeter cell across all stress periods
    n_cp = len(cp_cells)
    local_R = np.zeros((nper, n_cp))
    for p in range(nper):
        for i in range(n_cp):
            r_cp, c_cp = cp_cells[i]
            local_R[p, i] = 0.0 - h_data[p, 0, r_cp, c_cp]
            
    h_file.close()
    return idx, local_R

######################
### code execution ###
######################


if __name__ == "__main__":

    root = input("Enter root name for this run: ").strip()
    workspace = os.path.join(workspace_base, root)
    workspace = os.path.abspath(workspace)

    # define perimeter cells (control points) along the edges of the grid
    r_start, r_end, c_start, c_end = 5, 45, 5, 45
    cp_cells = []
    for c in range(c_start, c_end + 1): cp_cells.extend([(r_start, c), (r_end, c)])
    for r in range(r_start + 1, r_end): cp_cells.extend([(r, c_start), (r, c_end)])
    n_cp = len(cp_cells)
    print(f"Defined {nrow} x {ncol} grid with {n_cp} perimeter cells...")

    # define phantom wells at the same locations as the perimeter cells
    phantom_locs = cp_cells 
    n_phantom = len(phantom_locs)
    print(f"Set up {n_phantom} Phantom Wells along the perimeter...")

    # define number of real wells to simulate
    n_real = 10
    inner_rows = range(r_start + 1, r_end)
    inner_cols = range(c_start + 1, c_end)
    inner_coords = [(r, c) for r in inner_rows for c in inner_cols]
    
    # pick n_real random locations for the real wells from the inner grid (excluding perimeter)
    real_locs = random.sample(inner_coords, n_real)
    
    print(f"Running simulation for {n_real} Real wells...")
    real_wells_data = []
    for (r, c) in real_locs:
        real_wells_data.append({
            'r': r, 'c': c, 
            'Q': np.random.uniform(0.5, 2.0, nper)
        })
    
    # create a well stress period data structure for the real wells
    # random pumping rates between 0.5 and 2.0
    wel_spd_real = {p: [((0, w['r'], w['c']), -w['Q'][p]) for w in real_wells_data] for p in range(nper)}
    
    # run the "real" simulation to get the target drawdown at perimeter cells
    target_ws = os.path.join(workspace, "real")
    sim_t = setup_sim("real", wel_spd_real, target_ws)
    sim_t.write_simulation()
    sim_t.run_simulation(silent=True)
    
    # extract head data at perimeter cells for all stress periods to build the target vector 'b_target'
    hds_t_obj = flopy.utils.binaryfile.HeadFile(os.path.join(target_ws, "real.hds"))
    hds_t = hds_t_obj.get_alldata()
    hds_t_obj.close() 

    # b_target: the target drawdown at perimeter cells across all stress periods,
    b_target = np.array([0.0 - hds_t[p, 0, r, c] for p in range(nper) for r, c in cp_cells])

    # characterize the influence of each Phantom Well in parallel to build the surrogate model
    print(f"Characterizing {n_phantom} Phantom wells in parallel...")
    #  unpack (r, c) from phantom_locs (same as cp_cells)
    tasks = [(j, phantom_locs[j][0], phantom_locs[j][1], cp_cells, workspace) for j in range(n_phantom)]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_phantom_well, tasks))

    # reassemble the local response vectors into a global response matrix of shape (nper, n_cp, n_phantom)
    R_full = np.zeros((nper, n_cp, n_phantom))
    for idx, local_R in results:
        R_full[:, :, idx] = local_R

    ####################
    ### optimization ###
    ####################

    # assemble the global surrogate matrix 'G'
    # maps pumping rates at candidate wells to drawdown at perimeter cells
    print("Assembling global surrogate matrix...")
    R_shifted = np.zeros_like(R_full)
    R_shifted[1:, :, :] = R_full[:-1, :, :]
    U_inc_all = R_full - R_shifted

    # place the incremental response blocks into the global matrix 'G'
    # each block corresponds to the response of the system to pumping at a candidate well 
    # shifted according to the stress period
    # the resulting 'G' has dimensions (nper * n_cp) x (nper * n_phantom) and is
    # structured to capture the cumulative effect of pumping over time
    G = np.zeros((nper * n_cp, nper * n_phantom))
    for pp in range(nper):
        # Fills vertical blocks of G for each start-time 'pp'
        block = U_inc_all[:nper-pp, :, :].reshape(-1, n_phantom)
        G[pp*n_cp:, pp*n_phantom:(pp+1)*n_phantom] = block

    # impose L2 regularization to stabilize the optimization
    # prevent overfittings to noise in the surrogate model
    alpha = 0.01 
    G_reg = np.vstack([G, np.eye(nper * n_phantom) * alpha])
    b_reg = np.concatenate([b_target, np.zeros(nper * n_phantom)])
    
    # solve the regularized least squares problem to find the optimal pumping rates 'q_opt'
    # one per candidate well, showing how pumping rates evolve over the stress periods
    # bounds ensure that pumping rates are non-negative and do not exceed a reasonable maximum (e.g., 500)
    # lsq_linear optimization minimizes the difference between the surrogate model's predicted drawdown 
    # and the target drawdown at the perimeter cells
    # the result 'q_opt' is reshaped to have dimensions (nper, n_phantom) 
    # easier interpretation of pumping rates across stress periods and candidate wells
    print(f"Solving for {nper*n_phantom} optimized pumping rates... this may take a while...")
    res = lsq_linear(G_reg, b_reg, bounds=(0, 500))
    q_opt = res.x.reshape((nper, n_phantom))
    print("Success! Optimization complete. Proceeding to calculate surrogate match and save simulation results...")

    # calculate the surrogate's predicted drawdown at the perimeter cells using the optimized pumping rates
    # multiply the global surrogate matrix 'G' by the optimized pumping vector 'q_opt' 
    # returns the predicted drawdown 'b_match'
    b_match = G @ q_opt.flatten()
    
    #######################
    ### process results ###
    #######################

    # reshape the target drawdown and surrogate match back to (nper, n_cp) 
    # facilitates comparison and visualization
    match_reshaped = b_match.reshape((nper, n_cp))
    target_reshaped = b_target.reshape((nper, n_cp))

    # read script directory
    script_dir = Path(__file__).parent

    # navigate to results directory
    results_path = script_dir.parent / "results"

    # check the results directory exists
    results_path.mkdir(parents=True, exist_ok=True)

    # save results
    pd.DataFrame(target_reshaped).to_csv(results_path / f"{root}_target_drawdown.csv", index_label="SP")
    pd.DataFrame(q_opt).to_csv(results_path / f"{root}_optimized_pumping.csv", index_label="SP")
    pd.DataFrame(match_reshaped).to_csv(results_path / f"{root}_surrogate_match.csv", index_label="SP")

    print(f"Results saved to: {results_path}")
