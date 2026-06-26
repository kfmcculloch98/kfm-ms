import os
import shutil
from pathlib import Path
import numpy as np
import pyemu

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PEST_DIR = r"C:\Python\Personal\kfm-ms\codes\pest"
IES_EXE = r"C:\Python\Personal\kfm-ms\codes\binaries\PESTPP\windows\pestpp-ies.exe"
WORKER_ROOT = r"C:\Python\Personal\kfm-ms\codes\pest_workers"


def run_parallel_inversion(level):
    pst_name = f"inversion_level_{level}.pst"
    print(f"\nInitializing parallel PESTPP-IES run for Level {level}...")

    # Wipe out and regenerate worker tracking space cleanly
    if os.path.exists(WORKER_ROOT):
        try:
            shutil.rmtree(WORKER_ROOT)
        except OSError:
            os.system(f'rmdir /s /q "{WORKER_ROOT}"')

    os.makedirs(WORKER_ROOT, exist_ok=True)
    os.chdir(PEST_DIR)

    # Load pristine control file structures built by control.py
    pst = pyemu.Pst(pst_name)
    pst.control_data.noptmax = 2

    # Clean out autogen options exactly like your working model setup
    if "ies_autogen_par_ensem" in pst.pestpp_options:
        del pst.pestpp_options["ies_autogen_par_ensem"]
    if "ies_autogen_obs_ensem" in pst.pestpp_options:
        del pst.pestpp_options["ies_autogen_obs_ensem"]
    if "ies_autogen_par_ensem" in pst.control_data.formatted_values:
        del pst.control_data.formatted_values["ies_autogen_par_ensem"]

    num_reals = 100
    pe_filename = f"inversion_level_{level}.par.csv"

    print(f"Drawing parameter ensemble realizations via PyEMU bounds...")
    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst=pst, num_reals=num_reals)

    # Safely pull the pristine adjustable keys from the active tracking index
    active_pars = pst.parameter_data.loc[
        pst.parameter_data["partrans"] != "fixed"
    ].index.tolist()

    rng = np.random.default_rng(seed=42)
    for par_name in active_pars:
        base_val = float(pst.parameter_data.loc[par_name, "parval1"])
        random_noise = rng.uniform(-5.0, 5.0, size=num_reals)
        pe.loc[:, par_name] = np.clip(base_val + random_noise, 0.0, 150.0)

    # Save tracking dataset configurations
    pe.to_csv(pe_filename)
    pst.pestpp_options["ies_parameter_ensemble"] = pe_filename

    # Re-save cleanly to preserve Version 2 Block format rules
    pst.write(pst_name, version=2)

    # Fire workers using stable absolute paths matching your working model
    pyemu.utils.start_workers(
        worker_dir=PEST_DIR,
        exe_rel_path=IES_EXE,
        pst_rel_path=pst_name,
        num_workers=16,
        master_dir="master_run",
        worker_root=WORKER_ROOT,
        port=25318
    )

if __name__ == "__main__":
    choice = input("Enter inversion level to execute (1, 2, or 3): ").strip()
    if choice in ["1", "2", "3"]:
        run_parallel_inversion(choice)
    else:
        raise ValueError("Choice must be one of 1, 2, or 3")
