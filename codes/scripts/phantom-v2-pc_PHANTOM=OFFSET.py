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


# configure modflow and workspace
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


# simulation setup
nper = 52       # 52 weeks in a year
perlen = 7.0    # 7 days per week
nstp = 1        # 1 time step per week
tsmult = 1.0    # Equal step lengths
tdis_rc = [(perlen, nstp, tsmult) for _ in range(nper)]

cell_dim, ncol, nrow = 30, 92, 116 # cell size and number of rows and columns in the grid
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
    
    flopy.mf6.ModflowTdis(
        sim, 
        time_units="DAYS", 
        nper=nper, 
        perioddata=tdis_rc
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
    
    # All grid cells active to accommodate river borders
    idomain = np.ones((1, nrow, ncol), dtype=int)
    
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        top=2377,
        botm=1707,
        delr=cell_dim,
        delc=cell_dim,
        xorigin=0.0,
        yorigin=0.0,
        idomain=idomain,
    )

    # assign realistic hydraulic and storage properties based on bedrock aquifer characteristics
    flopy.mf6.ModflowGwfnpf(gwf, k=0.03, icelltype=0)
    flopy.mf6.ModflowGwfsto(gwf, ss=5e-4, transient=True)

    # set initial heads using a left-to-right hydraulic gradient
    strt_gradient = np.linspace(2377.0, 1707.0, ncol)
    strt_array = np.zeros((1, nrow, ncol), dtype=float)
    for c in range(ncol):
        strt_array[0, :, c] = strt_gradient[c]

    flopy.mf6.ModflowGwfic(gwf, strt=strt_array)

    # apply general head boundaries on left and right boundaries to maintain the initial gradient and allow for dynamic exchange of water with the surroundings
    ghb_data = []
    boundary_conductance = 0.05  

    # left boundary (column 0) - regional constant head source
    for r in range(nrow):
        ghb_data.append(((0, r, 0), strt_gradient[0], boundary_conductance))

    # right boundary (column ncol - 1) - regional constant head sink
    for r in range(nrow):
        ghb_data.append(((0, r, ncol - 1), strt_gradient[-1], boundary_conductance))

    ghb_period_data = {per: ghb_data for per in range(nper)}

    # add ghb package
    flopy.mf6.ModflowGwfghb(gwf, pname="ghb", stress_period_data=ghb_period_data)

    # add well package
    flopy.mf6.ModflowGwfwel(gwf, pname="wel", stress_period_data=wel_data)

    # configure output control
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        saverecord=[("head", "all")],
    )

    return sim

# PHANTOM parallel processing setup
def run_phantom_well(task_info):
    """
    worker function used to characterise the influence of a single phantom well.
    """
    # unpack strt_gradient from the task_info tuple
    idx, wr, wc, cp_cells, workspace, strt_gradient = task_info

    # stagger start times to reduce file i/o conflicts
    time.sleep(random.random() * 2)

    # create a unique workspace for each phantom well
    phantom_ws = os.path.join(workspace, f"phantom_{idx}")
    os.makedirs(phantom_ws, exist_ok=True)

    name = f"phantom_{idx}"

    # apply a scaled unit pumping rate at the phantom well for all stress periods
    scale_rate = 1000
    wel_spd = {p: [((0, wr, wc), -scale_rate)] for p in range(nper)}

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
            # drawdown = initial head - current head, where initial head is based on the left-to-right gradient
            h_init = strt_gradient[c_cp]
            local_R[p, i] = (h_init - h_data[p, 0, r_cp, c_cp]) / scale_rate 
            # normalised by the applied pumping rate to get a unit response

    h_file.close()
    return idx, local_R

# code execution
if __name__ == "__main__":

    root = input("Enter root name for this run: ").strip()
    workspace = os.path.abspath(os.path.join(workspace_base, root))

    # define perimeter control points with a fixed margin from the domain boundary
    margin = 25 # number of rows/columns to exclude from the perimeter control points
    
    # separation buffer between compliance perimeter and the real/phantom wells
    well_buffer = 5 # number of rows/columns to exclude from the real/phantom well locations to ensure they don't overlap with the perimeter control points

    print(f"Defined perimeter control points with a margin of {margin} cells...")
    print(f"Defined separation buffer between control points and wells of {well_buffer} cells...")

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

    # re-create the initial head vector
    strt_gradient = np.linspace(2377.0, 1707.0, ncol)

    # define the number of real wells to simulate
    n_real = 20
    
    # rows and columns are now restricted by the well_buffer
    inner_rows = range(r_start + well_buffer, r_end - well_buffer + 1)
    inner_cols = range(c_start + well_buffer, c_end - well_buffer + 1)
    inner_coords = [(r, c) for r in inner_rows for c in inner_cols]

    # check that the grid space isn't completely choked out by the buffer
    if len(inner_coords) < n_real:
        raise ValueError(f"Grid too small! well_buffer={well_buffer} left only {len(inner_coords)} cells for {n_real} wells.")

    # define a random seed for reproducibility
    random.seed(42)

    # select random locations for the real wells from the inner grid
    real_locs = random.sample(inner_coords, n_real)
    print(f"Real well locations (row, col): {real_locs}")

    # define a ratio of phantom wells to real wells
    phantom_ratio = 2
    n_phantom = math.floor(n_real * phantom_ratio) # round to nearest integer

    # calculate how many phantoms to generate per real well
    # e.g., if ratio = 1.50 and n_real = 20, phantoms_per_well = 1.5
    phantoms_per_well = n_phantom / n_real

    # define possible directions for phantom wells to be placed around each real well (8 surrounding cells)
    directions = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]

    phantom_locs = []
    for i, (r, c) in enumerate(real_locs):
        # determine if this specific well gets 1, 2, or more phantoms
        # distributes the extra phantoms evenly across the list
        count_for_this_well = math.floor((i + 1) * phantoms_per_well) - math.floor(i * phantoms_per_well)
        
        # pick distinct directions for this specific well so they don't overlap on the exact same cell
        chosen_dirs = random.sample(directions, count_for_this_well)
        
        for dr, dc in chosen_dirs:
            phantom_locs.append((r + dr, c + dc))

    print(f"Set up {len(phantom_locs)} PHANTOM wells...")


    print(f"Running simulation for {n_real} real wells...")

    # initialize a random number generator with a fixed seed
    rng = np.random.default_rng(seed=42)

    # generate pumping schedules for the real wells using a random uniform baseline rate multiplied by a random binary on/off pattern
    real_wells_data = []
    for r, c in real_locs:
        # pick a single baseline rate for this specific well
        base_rate = rng.uniform(273, 2730)
        
        # multiply that single rate by a 52-week array of 0s and 1s
        rates = base_rate * rng.choice([0, 1], size=nper)
        
        real_wells_data.append({
            "r": r,
            "c": c,
            "Q": rates
        })

    # assemble stress period data for the real wells
    wel_spd_real = {
        p: [((0, w["r"], w["c"]), -w["Q"][p]) for w in real_wells_data]
        for p in range(nper)
    }

    # save the real well schedule for PEST
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

    # write master truth file for PEST
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

    # calculate the target drawdown vector at the perimeter control points
    # based on the initial head gradient and the simulated heads from the reference simulation
    b_target = np.array(
        [strt_gradient[c] - hds_t[p, 0, r, c] for p in range(nper) for r, c in cp_cells]
    )

    # characterise the response of each phantom well in parallel
    print(f"Characterising {n_phantom} PHANTOM wells in parallel...")
    tasks = [
        (j, phantom_locs[j][0], phantom_locs[j][1], cp_cells, workspace, strt_gradient)
        for j in range(n_phantom)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_phantom_well, tasks))

    # assemble the response tensor
    R_full = np.zeros((nper, n_cp, n_phantom))
    for idx, local_R in results:
        R_full[:, :, idx] = local_R

    # assemble the global surrogate matrix (G) using index matching to ensure perfect alignment with b_target
    print("Assembling global surrogate matrix (G) using index matching...")
    
    # G maps optimized rates (nper * n_phantom) to target drawdowns (nper * n_cp)
    G = np.zeros((nper * n_cp, nper * n_phantom))
    
    # map out the global row indexing system to match b_target exactly
    # row dimension: (Time * n_cp) + CP_index
    for p_obs in range(nper):
        for i_cp in range(n_cp):
            global_row = (p_obs * n_cp) + i_cp
            
            # map out the global column indexing system for the causal pumping stresses
            # column dimension: (Time * n_phantom) + Phantom_index
            for p_pump in range(nper):
                # check if the pumping pulse at p_pump would have influenced the drawdown at p_obs
                if p_obs >= p_pump:
                    # calculate how many weeks have elapsed since this specific pumping pulse began
                    elapsed_weeks = p_obs - p_pump
                    
                    for j_phan in range(n_phantom):
                        global_col = (p_pump * n_phantom) + j_phan
                        
                        # extract the unit impact coefficient from the parallel response data
                        unit_coefficient = R_full[elapsed_weeks, i_cp, j_phan]
                        
                        # place it perfectly inside the Toeplitz matrix
                        G[global_row, global_col] = unit_coefficient

    # solve for the optimized pumping rates using the raw index-aligned matrix
    print(f"Solving for {nper * n_phantom} optimised pumping rates...")
    res = lsq_linear(G, b_target, bounds=(0, 3000))
    q_opt = res.x.reshape((nper, n_phantom))
    print("Success! Optimization complete. Proceeding to calculate surrogate-predicted drawdown at the perimeter control points...")

    # calculate the surrogate-predicted drawdown at the perimeter control points
    b_match = G @ q_opt.flatten()

    # process results
    match_reshaped = b_match.reshape((nper, n_cp))
    target_reshaped = b_target.reshape((nper, n_cp))

    # save results in the project results folder
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

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
    np.save(results_path / "G_real_basis.npy", G)

    print(f"Results saved to: {results_path}")