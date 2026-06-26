import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyemu

# ==============================================================================
# PATH DEFINITIONS (Enforcing Path Object Safety)
# ==============================================================================
BASE_DIR = Path(r"C:\Python\Personal\kfm-ms\codes")
PEST_DIR = BASE_DIR / "pest"
IES_EXE = BASE_DIR / "binaries" / "PESTPP" / "windows" / "pestpp-ies.exe"
WORKER_ROOT = BASE_DIR / "pest_workers"
MASTER_DIR = PEST_DIR / "master_run"  # Isolated master directory path

def run_parallel_inversion(level):
    pst_name = f"inversion_level_{level}.pst"
    print(f"\n[INFO] Initializing parallel PESTPP-IES run for Level {level}...")

    # Clear out old worker and master directories cleanly
    for target_dir in [WORKER_ROOT, MASTER_DIR]:
        if target_dir.exists():
            print(f"[INFO] Cleaning existing directory: {target_dir}")
            try:
                shutil.rmtree(target_dir)
            except OSError:
                os.system(f'rmdir /s /q "{target_dir}"')
                
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Change context to PEST_DIR so pyemu reads relative file structures correctly
    os.chdir(str(PEST_DIR))
    
    # Load the control file parameters
    pst = pyemu.Pst(pst_name)
    pst.control_data.noptmax = 2  # Run for 2 optimization iteration cycles
    
    # Pop autogen flags cleanly from options without disrupting boundary configurations
    pst.pestpp_options.pop("ies_autogen_par_ensem", None)
    pst.pestpp_options.pop("ies_autogen_obs_ensem", None)
    pst.pestpp_options.pop("ies_par_en_std_dev", None) 
        
    NUM_REALIZATIONS = 250
    pst.pestpp_options["ies_num_reals"] = NUM_REALIZATIONS
    
    # Re-enforce the blind bounds strategy right here to protect against upstream overrides
    pst.pestpp_options["ies_enforce_bounds"] = "false" # Allows PEST++ to climb past 0.50
        
    pe_filename = f"inversion_level_{level}.par.csv"
    print(f"[INFO] Generating high-diversity log-uniform parameter ensemble ({NUM_REALIZATIONS} entries)...")
    
    # Rebuild a manually controlled, log-normally distributed parameter ensemble
    par_df = pst.parameter_data.copy()
    num_pars = len(par_df)
    
    # Pre-allocate ensemble array matrix layout
    ensemble_matrix = np.zeros((NUM_REALIZATIONS, num_pars))
    
    for idx, (p_name, row) in enumerate(par_df.iterrows()):
        # Pull low and high boundaries securely (They are now 0.05 and 0.50!)
        lb = max(row["parlbnd"], 1e-4) 
        ub = row["parubnd"]
        
        # Draw log-uniformly across your tightly constrained out-of-distribution limits
        log_draws = np.random.uniform(np.log10(lb), np.log10(ub), size=NUM_REALIZATIONS)
        ensemble_matrix[:, idx] = 10**log_draws

    # Wrap the matrix into a formal PyEMU ParameterEnsemble object structure
    pe = pyemu.ParameterEnsemble(pst=pst, df=pd.DataFrame(ensemble_matrix, columns=par_df.index))
    pe.to_csv(pe_filename)
    
    pst.pestpp_options["ies_parameter_ensemble"] = pe_filename
    pst.model_command = "python forward_run.py"

    # Purge any residual ies_ keys from the formatted_values dictionary to prevent conflicts 
    keys_to_purge = [k for k in pst.control_data.formatted_values.keys() if "ies_" in k.lower()]
    for k in keys_to_purge:
        del pst.control_data.formatted_values[k]

    # Re-save updated master configurations using Version 2 Format cleanly
    pst.write(pst_name, version=2)

    # Ensure a copy of the executable sits in the PEST directory for relative tracking
    local_exe_name = IES_EXE.name
    local_exe_path = PEST_DIR / local_exe_name
    if not local_exe_path.exists():
        shutil.copy(IES_EXE, local_exe_path)

    print(f"[INFO] Starting master manager run and spinning up 16 background workers...")
    
    # Route template tracking explicitly using concrete string transformations
    pyemu.utils.start_workers(
        worker_dir=str(PEST_DIR),         # Source template directoryc
        exe_rel_path=local_exe_name,      # Executable name relative to worker_dir
        pst_rel_path=pst_name,            # Control file name relative to worker_dir
        num_workers=16,
        master_dir=str(MASTER_DIR),       # Absolute path prevents local circular cloning
        worker_root=str(WORKER_ROOT),     # Target path where 16 worker folders sit
        port=25318,
        cleanup=True                      # Auto-drops temporary metadata hooks
    )
    
if __name__ == "__main__":
    choice = input("Enter inversion level to execute (1, 2, or 3): ").strip()
    if choice in ["1", "2", "3"]:
        run_parallel_inversion(choice)
