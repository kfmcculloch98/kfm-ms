import os
from ipykernel.iostream import MASTER
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
    # Use mean values of pumping rates across all stress periods to create a flat template for the inversion

    # Load the master truth CSV file from the source model workspace
    source_model_ws = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "sims", root_name, "real"))
    template_truth_csv = os.path.join(source_model_ws, "master_truth.csv")
    
    print(f"[+] Overwriting template values with means...")
    df_temp = pd.read_csv(template_truth_csv)
    
    # Calculate the true temporal mean for each individual well across stress periods
    # This preserves spatial separation but completely flattens out time-series variations
    well_means = df_temp.groupby("well_id")["q"].mean().to_dict()
    
    # Inject the static mean values back into the parameter column
    df_temp["q"] = df_temp["well_id"].map(well_means)
    
    # Save the flattened mean file back to the sims folder
    df_temp.to_csv(template_truth_csv, index=False)
    print(f"[+] Success. Template file 'q' column is now locked to flat uniform means.")

    pf = pyemu.utils.PstFrom(
        original_d=source_model_ws,
        new_d=PEST_DIR,
        remove_existing=True,
        longnames=True
    )

    # Copy the surrogate matrix to the master template folder
    source_g = os.path.join(RESULTS_DIR, "G_real_basis.npy")
    dest_g = os.path.join(PEST_DIR, "G_real_basis.npy")
    shutil.copy(source_g, dest_g)

    # Register parameters for the inversion using the master truth file
    pf.add_parameters(
        filenames=MASTER_TRUTH_FILE,
        par_type="grid",
        index_cols=["r", "c", "sp"],
        use_cols=["q"],
        transform="none"  # Protects against log(0) errors during uniform draws
    )

    # Copy the observation file structure
    source_obs = os.path.join(source_model_ws, "obs.csv")
    dest_dummy = os.path.join(PEST_DIR, "dummy_obs.csv")
    shutil.copy(source_obs, dest_dummy) # Active layout template source for plotting
    dest_obs = os.path.join(PEST_DIR, "obs.csv")
    shutil.copy(source_obs, dest_obs) # Initial target source for PstFrom generation mapping

    # Add observations using the copied file structure
    pf.add_observations("obs.csv", index_cols=["obsnme"], use_cols=["obsval"])

    # Register your surrogate code execution hook
    pf.add_py_function("surrogate.py", call_str="run_surrogate()", is_pre_cmd=False)

    # Compile the control file
    pst = pf.build_pst()

    # Mute weights of near-zero observations to avoid skewing the inversion
    low_signal_mask = pst.observation_data["obsval"] < 0.1 # 10 cm threshold
    pst.observation_data.loc[low_signal_mask, "weight"] = 0.0

    # Define execution commands
    pst.model_command = f'"{PYTHON_LOC}" forward_run.py'
    pst.control_data.noptmax = 0

    # Set parameters 
    pst.parameter_data.loc[:, "partrans"] = "none"
    pst.parameter_data.loc[:, "parval1"] = 1.0  # multiplier for pumping rates  
    pst.parameter_data.loc[:, "parlbnd"] = 0.05  # Lower bound multiplier floor
    pst.parameter_data.loc[:, "parubnd"] = 5.0  # Upper bound multiplier cap

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

    # Set PEST++ options for the inversion
    pst.pestpp_options["ies_num_reals"] = 250  # Number of realizations for the inversion
    pst.pestpp_options["ies_autogen_par_ensem"] = "off" # Turn off to prioritize your log-uniform draw
    pst.pestpp_options["ies_enforce_bounds"] = "false" # Allows PEST++ to break out past 0.50 to discover the peaks!

    final_pst_path = os.path.join(PEST_DIR, f"inversion_level_{level}.pst")
    pst.write(final_pst_path, version=2)
    print(f"\n[!] Success: Control file created: inversion_level_{level}.pst")

if __name__ == "__main__":
    root_name = input("Enter root name for this run (e.g. realistic): ").strip()
    choice = input("Select level (1, 2, or 3): ").strip()
    if choice in ["1", "2", "3"]:
        build_modular_pst(choice, root_name)
