import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyemu

print("[SCRIPT] run_ies-pc.py loaded")
print(f"[SCRIPT] Python: {sys.executable}")
print(f"[SCRIPT] CWD at start: {os.getcwd()}")

BASE_DIR = Path(r"C:\Python\Personal\kfm-ms\codes")
PEST_DIR = BASE_DIR / "pest_level_1"
IES_EXE = BASE_DIR / "binaries" / "PESTPP" / "windows" / "pestpp-ies.exe"
WORKER_ROOT = BASE_DIR / "pest_workers"
MASTER_DIR = BASE_DIR / "master_run"
SCRIPTS_DIR = BASE_DIR / "scripts"

PYTHON_EXE = sys.executable


def run_parallel_inversion(level, nreals=250, noptmax=2, num_workers=16, cleanup=False):
    level = str(level).strip()
    pst_name = f"inversion_level_{level}.pst"

    print(f"\n[INFO] Initializing parallel PESTPP-IES run for Level {level}...")
    print(f"[INFO] nreals = {nreals}")
    print(f"[INFO] noptmax = {noptmax}")
    print(f"[INFO] num_workers = {num_workers}")
    print(f"[INFO] cleanup = {cleanup}")
    print(f"[INFO] PEST_DIR: {PEST_DIR}")
    print(f"[INFO] MASTER_DIR: {MASTER_DIR}")
    print(f"[INFO] WORKER_ROOT: {WORKER_ROOT}")
    print(f"[INFO] IES_EXE: {IES_EXE}")
    print(f"[INFO] PST name: {pst_name}")

    pst_path = PEST_DIR / pst_name
    sur_path = SCRIPTS_DIR / "surrogate.py"

    for path in [pst_path, IES_EXE, sur_path]:
        print(f"[CHECK] {path} -> {'EXISTS' if path.exists() else 'MISSING'}")

    if not pst_path.exists():
        raise FileNotFoundError(f"Missing PST file: {pst_path}")
    if not IES_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {IES_EXE}")
    if not sur_path.exists():
        raise FileNotFoundError(f"Missing script: {sur_path}")

    if cleanup:
        for target_dir in [WORKER_ROOT, MASTER_DIR]:
            if target_dir.exists():
                print(f"[INFO] Cleaning existing directory: {target_dir}")
                shutil.rmtree(target_dir, ignore_errors=True)

    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    os.chdir(PEST_DIR)
    print(f"[INFO] Changed working directory to: {os.getcwd()}")

    pst = pyemu.Pst(pst_name)
    pst.control_data.noptmax = noptmax
    pst.pestpp_options["ies_num_reals"] = int(nreals)

    pst.model_command = f'"{PYTHON_EXE}" forward_run.py'
    print(f"[INFO] model_command set to: {pst.model_command}")

    # Build ensemble
    par_df = pst.parameter_data.copy()
    print(f"[INFO] Number of parameters: {len(par_df)}")

    ens = np.zeros((nreals, len(par_df)))

    for idx, (p_name, row) in enumerate(par_df.iterrows()):
        partrans = str(row["partrans"]).strip().lower()

        if partrans == "fixed":
            ens[:, idx] = float(row["parval1"])
        else:
            lb = float(row["parlbnd"])
            ub = float(row["parubnd"])

            if not np.isfinite(lb) or not np.isfinite(ub):
                raise ValueError(f"Non-finite bounds for parameter {p_name}: lb={lb}, ub={ub}")
            if ub <= lb:
                raise ValueError(f"Invalid bounds for parameter {p_name}: lb={lb}, ub={ub}")

            ens[:, idx] = np.random.uniform(lb, ub, size=nreals)

    # df = pd.DataFrame(ens, columns=par_df.index)
    # print("[INFO] Ensemble preview:")
    # print(df.head())

    # pe = pyemu.ParameterEnsemble.from_dataframe(pst, df)

    # pe_filename = f"prior_ens_level_{level}.csv"
    # pe_path = PEST_DIR / pe_filename
    # pe.to_csv(pe_path)
    # print(f"[INFO] Wrote ensemble file: {pe_path}")

    # pst.pestpp_options["ies_parameter_ensemble"] = pe_filename

    # Remove stale options if present
    pst.pestpp_options.pop("ies_autogen_par_ensem", None)
    pst.pestpp_options.pop("ies_autogen_obs_ensem", None)
    pst.pestpp_options.pop("ies_par_en_std_dev", None)

    pst.write(pst_name, version=2)
    print(f"[INFO] Wrote control file: {PEST_DIR / pst_name}")

    local_exe_path = PEST_DIR / IES_EXE.name
    if not local_exe_path.exists():
        print(f"[INFO] Copying executable to: {local_exe_path}")
        shutil.copy(IES_EXE, local_exe_path)

    # Stage only surrogate.py; do NOT overwrite PstFrom-generated forward_run.py
    print("[INFO] Copying surrogate.py into PEST_DIR...")
    shutil.copy(sur_path, PEST_DIR / "surrogate.py")
    print("[INFO] Leaving PstFrom-generated forward_run.py in place.")

    print("[INFO] Files now in PEST_DIR:")
    for f in sorted(os.listdir(PEST_DIR)):
        print("   ", f)

    print(f"[INFO] Starting workers with {num_workers} workers...")
    pyemu.utils.start_workers(
        worker_dir=str(PEST_DIR),
        exe_rel_path=IES_EXE.name,
        pst_rel_path=pst_name,
        num_workers=num_workers,
        master_dir=str(MASTER_DIR),
        worker_root=str(WORKER_ROOT),
        port=25318,
        cleanup=cleanup
    )


if __name__ == "__main__":
    print("[SCRIPT] Entered __main__")
    choice = input("Enter inversion level to execute (1, 2, or 3): ").strip()

    if choice in ["1", "2", "3"]:
        run_parallel_inversion(
            level=choice,
            nreals=250,
            noptmax=2,
            num_workers=16,
            cleanup=False
        )
    else:
        raise ValueError("Level must be 1, 2, or 3")