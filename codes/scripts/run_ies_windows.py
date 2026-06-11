import os
import shutil
import pyemu
from pathlib import Path

# paths
PEST_DIR = r"C:\Python\Personal\kfm-ms\codes\pest"
IES_EXE = r"C:\Python\Personal\kfm-ms\codes\binaries\PESTPP\windows\pestpp-ies.exe"
PST_NAME = "inversion_level_1.pst"

# create a sibling directory for each parallel execution
WORKER_ROOT = r"C:\Python\Personal\kfm-ms\codes\pest_workers"

def run_parallel_inversion():
    print("Initializing parallel PESTPP-IES run via pyemu...")

    # delete any old worker directories
    if os.path.exists(WORKER_ROOT):
        print(f"Cleaning up previous workspace at {WORKER_ROOT}...")
        try:
            shutil.rmtree(WORKER_ROOT)
        except OSError:
            os.system(f'rmdir /s /q "{WORKER_ROOT}"')
            
    os.makedirs(WORKER_ROOT, exist_ok=True)

    # change local execution context to PEST source directory
    os.chdir(PEST_DIR)
    
    print(f"Loading {PST_NAME} to inject ensemble-generation rules...")
    pst = pyemu.Pst(PST_NAME)
    
    # number of iterations is set low
    pst.control_data.noptmax = 2
    
    # prevent pestpp from auto-generating ensembles on its own since we'll handle that via pyemu's built-in functionality
    if "ies_autogen_par_ensem" in pst.pestpp_options:
        del pst.pestpp_options["ies_autogen_par_ensem"]
    if "ies_autogen_obs_ensem" in pst.pestpp_options:
        del pst.pestpp_options["ies_autogen_obs_ensem"]
        
    # strip the double quotes from the command string to prevent pestpp from choking on them
    raw_cmd = pst.model_command[0]
    pst.model_command = [raw_cmd.replace('"', '')]
    
    # re-write the baseline control file
    pst.write(PST_NAME, version=2)
    
    # force PEST to accept the commands in its own isolated section block
    options_block = [
        "\n",
        "* pestpp options\n",
        "ies_autogen_par_ensem          true\n",
        "ies_autogen_obs_ensem          true\n",
        "ies_ensemble_size             300\n"
    ]

    with open(PST_NAME, "a") as f:
        f.writelines(options_block)
    # =================================================================
    
    print(f"Spawning 8 local workers inside {WORKER_ROOT}...")
    pyemu.utils.start_workers(
        worker_dir=PEST_DIR,           # source folder to duplicate
        exe_rel_path=IES_EXE,          # PEST binary path
        pst_rel_path=PST_NAME,         # PEST control file name
        num_workers=8,                 # number of parallel local workers to spawn
        master_dir="master_run",       # master coordination folder name
        worker_root=WORKER_ROOT,       # isolated parent directory for all run folders
        port=25318                     # TCP port
    )
    
if __name__ == "__main__":
    run_parallel_inversion()