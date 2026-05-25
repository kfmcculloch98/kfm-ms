import os
import shutil
import pyemu
from pathlib import Path

# paths
HOME = Path.home() # Dynamically points to /home/u30/kfmcculloch
PEST_DIR = os.path.abspath(HOME / "kfm-ms/codes/pest")
IES_EXE = os.path.abspath(HOME / "kfm-ms/codes/binaries/PESTPP/linux/pestpp-ies")
PST_NAME = "inversion_level_1.pst"
WORKER_ROOT = os.path.abspath(HOME / "kfm-ms/codes/pestpp/pest_workers")

def run_parallel_inversion():
    print("Initializing parallel PESTPP-IES run via pyemu...")

    # delete any old worker directories
    if os.path.exists(WORKER_ROOT):
        print(f"Cleaning up previous workspace at {WORKER_ROOT}...")
        shutil.rmtree(WORKER_ROOT)
    os.makedirs(WORKER_ROOT, exist_ok=True)

    # change local execution context to PEST source directory
    os.chdir(PEST_DIR)

    pyemu.utils.start_workers(
        worker_dir=PEST_DIR,           # source folder to duplicate
        exe_rel_path=IES_EXE,          # PEST binary path
        pst_rel_path=PST_NAME,         # PEST control file name
        num_workers=7,                 # number of parallel local workers to spawn (# of cores - 2)
        master_dir=os.path.join(WORKER_ROOT, "master_run"), # master subdirectory name
        worker_root=WORKER_ROOT,       # isolated parent directory for all run folders
        port=4005                      # TCP port
    )

if __name__ == "__main__":
    run_parallel_inversion()
