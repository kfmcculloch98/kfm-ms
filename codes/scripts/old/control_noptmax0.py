import os
import pandas as pd
import shutil
from pathlib import Path
import pyemu

# Configure paths precisely matching your working structure
SCRIPT_DIR = Path(__file__).resolve().parent
PEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "pest"))
RESULTS_DIR = Path(r"c:\Python\Personal\kfm-ms\codes\results")
PYTHON_LOC = Path(r"c:\Python\Personal\kfm-ms\.venv\Scripts\python.exe")
MASTER_TRUTH_FILE = "master_truth.csv"

def build_modular_pst(level, root_name):
    # Dynamically inject the root name back into your original path structure
    model_ws = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "sims", root_name, "real"))
    print(f"\n[+] Initializing pest setup in: {PEST_DIR}")
    print(f"[+] Source model workspace: {model_ws}")

    # Initialize the pest-from workflow exactly as you had it
    pf = pyemu.utils.PstFrom(
        original_d=model_ws,
        new_d=PEST_DIR,
        remove_existing=True,
        longnames=True
    )

    # Copy the surrogate matrix to the master template folder
    source_g = os.path.join(RESULTS_DIR, "G_real_basis.npy")
    dest_g = os.path.join(PEST_DIR, "G_real_basis.npy")
    shutil.copy(source_g, dest_g)

    # =====================================================================
    # ADDED: COPY THE BASELINE HEADS ARRAY FOR WORKER ITERATIONS
    # =====================================================================
    source_base = os.path.join(RESULTS_DIR, "b_baseline_flat.npy")
    dest_base = os.path.join(PEST_DIR, "b_baseline_flat.npy")
    if os.path.exists(source_base):
        shutil.copy(source_base, dest_base)
        print("[+] Successfully copied 'b_baseline_flat.npy' to PEST active context.")
    else:
        print(f"[-] Warning: 'b_baseline_flat.npy' missing from {RESULTS_DIR}")

    # Register parameters for the inversion using the master truth file
    pf.add_parameters(
        filenames="master_truth.csv",
        par_type="grid",
        index_cols=["r", "c", "sp"],
        use_cols=["q"],
        transform="none"  # Protects against log(0) errors during uniform draws
    )

    # =====================================================================
    # FIXED: PULL FROM LOCAL model_ws INSTEAD OF THE GLOBAL HARDCODED PATH
    # =====================================================================
    source_obs = os.path.join(model_ws, "obs.csv")
    dest_dummy = os.path.join(PEST_DIR, "dummy_obs.csv")
    shutil.copy(source_obs, dest_dummy)

    # Add observations using the copied file structure
    pf.add_observations("obs.csv", index_cols=["obsnme"], use_cols=["obsval"])

    # Register your surrogate code execution hook
    pf.add_py_function("surrogate.py", call_str="run_surrogate()", is_pre_cmd=False)

    # Compile the control file
    pst = pf.build_pst()

    # Define execution commands
    pst.model_command = f'"{PYTHON_LOC}" forward_run.py'
    pst.control_data.noptmax = 0

    # Set parameters 
    pst.parameter_data.loc[:, "partrans"] = "none"
    pst.parameter_data.loc[:, "parval1"] = 1.0  # multiplier for pumping rates  
    pst.parameter_data.loc[:, "parlbnd"] = 1e-4  # Lower bound to prevent log(0) issues
    pst.parameter_data.loc[:, "parubnd"] = 5 # Upper bound for pumping rates

    # Load master truth for filtering scenarios
    df_truth = pd.read_csv(os.path.join(PEST_DIR, MASTER_TRUTH_FILE))

    if level == "1":
        print("[+] You selected Level 1 (Full Knowledge)")
        active_wells = df_truth[df_truth["q"] > 0][["r", "c"]].drop_duplicates()
        active_tags = "r:" + active_wells["r"].astype(int).astype(str) + \
                      "_c:" + active_wells["c"].astype(int).astype(str) + "_"
        
        active_mask = pst.parameter_data.index.str.contains('|'.join(active_tags))
        pst.parameter_data.loc[active_mask, "partrans"] = "none"
        pst.parameter_data.loc[active_mask, "parval1"] = 1.0  

    elif level == "2":
        print("[+] You selected Level 2 (Known Locations)")
        unique_locs = df_truth[["r", "c"]].drop_duplicates()
        for _, row in unique_locs.iterrows():
            loc_tag = f"r:{int(row.r)}_c:{int(row.c)}"
            mask = pst.parameter_data.index.str.contains(loc_tag)
            pst.parameter_data.loc[mask, "partrans"] = "none"
            pst.parameter_data.loc[mask, "parval1"] = 1.0

    elif level == "3":
        print("[+] You selected Level 3 (Blind)")
        pst.parameter_data.loc[:, "partrans"] = "none"
        pst.parameter_data.loc[:, "parval1"] = 1.0

    pst.pestpp_options["ies_num_reals"] = 500  # Set the number of realizations for the inversion
    pst.pestpp_options["ies_autogen_par_ensem"] = "on" # Enable automatic ensemble generation for parameters
    pst.pestpp_options["ies_par_en_std_dev"] = 0.5 # Set the standard deviation for parameter ensemble generation
    pst.pestpp_options["ies_enforce_bounds"] = "true" # Enforce parameter bounds during the inversion process

    final_pst_path = os.path.join(PEST_DIR, f"inversion_level_{level}.pst")
    pst.write(final_pst_path, version=2)
    print(f"\n[!] Success: Control file created: inversion_level_{level}.pst")

if __name__ == "__main__":
    root_name = input("Enter root name for this run (e.g. realistic): ").strip()
    choice = input("Select level (1, 2, or 3): ").strip()
    if choice in ["1", "2", "3"]:
        build_modular_pst(choice, root_name)