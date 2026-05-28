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
    
    pyemu.utils.start_workers(
        worker_dir=PEST_DIR,           # source folder to duplicate
        exe_rel_path=IES_EXE,          # PEST binary path
        pst_rel_path=PST_NAME,         # PEST control file name
        num_workers=8,                 # number of parallel local workers to spawn
        master_dir="master_run",       # <-- FIX: Just the folder name string! PyEMU builds this inside worker_root
        worker_root=WORKER_ROOT,       # isolated parent directory for all run folders
        port=25318                     # TCP port
    )
    
if __name__ == "__main__":
    run_parallel_inversion()
