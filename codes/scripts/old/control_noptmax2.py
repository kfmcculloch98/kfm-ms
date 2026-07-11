import os
import shutil
from pathlib import Path
import pandas as pd
import pyemu
import numpy as np

# ==============================================================================
# PATH DEFINITIONS
# ==============================================================================
BASE_DIR = Path(r"C:\Python\Personal\kfm-ms\codes")
PEST_DIR = BASE_DIR / "pest"
IES_EXE = BASE_DIR / "binaries" / "PESTPP" / "windows" / "pestpp-ies.exe"
WORKER_ROOT = BASE_DIR / "pest_workers"
MASTER_DIR = PEST_DIR / "master_run"  # Isolated master directory path

MASTER_TRUTH_FILE = "master_truth.csv"
GRID_CANDIDATES_FILE = "grid_candidates.csv"
PARAM_ORDER_FILE = "param_order.csv"


def parse_parameter_locations(pst):
    """
    Add numeric r, c, sp columns to pst.parameter_data by parsing the
    actual PEST-generated parameter names.
    """
    pnames = pst.parameter_data.index.to_series().astype(str).str.lower()

    pst.parameter_data["r"] = pd.to_numeric(
        pnames.str.extract(r"r:(\d+)")[0],
        errors="coerce"
    )
    pst.parameter_data["c"] = pd.to_numeric(
        pnames.str.extract(r"c:(\d+)")[0],
        errors="coerce"
    )
    pst.parameter_data["sp"] = pd.to_numeric(
        pnames.str.extract(r"sp:(\d+)")[0],
        errors="coerce"
    )

    bad = pst.parameter_data[pst.parameter_data[["r", "c", "sp"]].isna().any(axis=1)]
    if len(bad) > 0:
        print(f"[WARN] {len(bad)} parameter names could not be parsed into r/c/sp.")
        print(bad.index[:10].tolist())
    else:
        print("[INFO] Parameter name parsing into r/c/sp completed successfully.")

    return pst


def run_parallel_inversion(level):
    pst_name = f"inversion_level_{level}.pst"
    print(f"\n[INFO] Initializing parallel PESTPP-IES run for Level {level}...")

    # -------------------------------------------------------------------------
    # Clean old worker and master directories
    # -------------------------------------------------------------------------
    for target_dir in [WORKER_ROOT, MASTER_DIR]:
        if target_dir.exists():
            print(f"[INFO] Cleaning existing directory: {target_dir}")
            try:
                shutil.rmtree(target_dir)
            except OSError:
                os.system(f'rmdir /s /q "{target_dir}"')

    WORKER_ROOT.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Ensure PEST_DIR is the working directory
    # -------------------------------------------------------------------------
    os.chdir(str(PEST_DIR))

    # -------------------------------------------------------------------------
    # Load control file
    # -------------------------------------------------------------------------
    pst = pyemu.Pst(pst_name)
    pst.control_data.noptmax = 2

    # Clean obsolete / conflicting options
    pst.pestpp_options.pop("ies_autogen_par_ensem", None)
    pst.pestpp_options.pop("ies_autogen_obs_ensem", None)
    pst.pestpp_options.pop("ies_par_en_std_dev", None)

    NUM_REALIZATIONS = 250
    pst.pestpp_options["ies_num_reals"] = NUM_REALIZATIONS

    # -------------------------------------------------------------------------
    # Make sure parameter locations are parsed before any filtering
    # -------------------------------------------------------------------------
    pst = parse_parameter_locations(pst)

    # -------------------------------------------------------------------------
    # Build a parameter ensemble
    # -------------------------------------------------------------------------
    pe_filename = f"inversion_level_{level}.par.csv"
    print(f"[INFO] Generating uniform parameter ensemble with {NUM_REALIZATIONS} realizations...")

    par_df = pst.parameter_data.copy()
    num_pars = len(par_df)

    ensemble_matrix = np.zeros((NUM_REALIZATIONS, num_pars))

    for idx, (p_name, row) in enumerate(par_df.iterrows()):
        if row["partrans"] == "fixed":
            ensemble_matrix[:, idx] = row["parval1"]
        else:
            lb = row["parlbnd"]
            ub = row["parubnd"]
            ensemble_matrix[:, idx] = np.random.uniform(lb, ub, size=NUM_REALIZATIONS)

    pe = pyemu.ParameterEnsemble(
        pst=pst,
        df=pd.DataFrame(ensemble_matrix, columns=par_df.index)
    )
    pe.to_csv(pe_filename)

    pst.pestpp_options["ies_parameter_ensemble"] = pe_filename

    # -------------------------------------------------------------------------
    # Model command
    # -------------------------------------------------------------------------
    pst.model_command = "python forward_run.py"

    # Clear out any lingering formatted values related to ies_
    keys_to_purge = [k for k in pst.control_data.formatted_values.keys() if "ies_" in k.lower()]
    for k in keys_to_purge:
        del pst.control_data.formatted_values[k]

    # Rewrite PST
    pst.write(pst_name, version=2)

    # -------------------------------------------------------------------------
    # Copy executable locally if needed
    # -------------------------------------------------------------------------
    local_exe_name = IES_EXE.name
    local_exe_path = PEST_DIR / local_exe_name
    if not local_exe_path.exists():
        shutil.copy(IES_EXE, local_exe_path)

    # -------------------------------------------------------------------------
    # Copy scripts into PEST directory
    # -------------------------------------------------------------------------
    print("[INFO] Staging fresh Python scripts into the template folder...")
    scripts_source = BASE_DIR / "scripts"
    shutil.copy(scripts_source / "forward_run.py", PEST_DIR / "forward_run.py")
    shutil.copy(scripts_source / "surrogate.py", PEST_DIR / "surrogate.py")

    # -------------------------------------------------------------------------
    # Read truth and determine concealment constraints
    # -------------------------------------------------------------------------
    truth_file = PEST_DIR / MASTER_TRUTH_FILE
    if not truth_file.exists():
        raise FileNotFoundError(f"Missing truth file: {truth_file}")

    df_truth = pd.read_csv(truth_file)
    sp_values = sorted(df_truth["sp"].dropna().unique().tolist())
    total_pumping_per_sp = df_truth.groupby("sp")["q"].sum().to_dict()

    true_locs = set(zip(df_truth["r"].astype(int), df_truth["c"].astype(int)))

    print(f"[INFO] True locations: {len(true_locs)}")
    print(f"[INFO] Stress periods in truth: {sp_values}")

    # -------------------------------------------------------------------------
    # Level-specific logic
    # -------------------------------------------------------------------------
    if str(level) == "1":
        print("[+] Level 1 Selected: Spatial Locations & Total Volume Known")

        # Active only if cell location is one of the true well locations
        active_mask = pst.parameter_data.apply(
            lambda row: (
                pd.notna(row["r"])
                and pd.notna(row["c"])
                and (int(row["r"]), int(row["c"])) in true_locs
            ),
            axis=1
        )

        pst.parameter_data.loc[~active_mask, "partrans"] = "fixed"
        pst.parameter_data.loc[~active_mask, "parval1"] = 0.0

        print(f"[DEBUG] Level 1 active parameters: {int(active_mask.sum())}")
        print(f"[DEBUG] Level 1 fixed parameters: {int((~active_mask).sum())}")

        for sp in sp_values:
            sp_mask = pst.parameter_data["sp"] == int(sp)
            sp_pars = pst.parameter_data.loc[sp_mask & active_mask]
            par_list = list(sp_pars.index)

            print(f"[DEBUG] Level 1, sp={sp}, matched active parameters={len(par_list)}")

            if len(par_list) == 0:
                raise ValueError(f"No active parameters matched for stress period {sp}")

            coef_dictionary = {par: 1.0 for par in par_list}

            pst.add_pi_equation(
                par_names=par_list,
                pilbl=f"tot_q_sp{int(sp)}".lower(),
                rhs=float(total_pumping_per_sp[sp]),
                weight=100.0,
                obs_group="pumping_constraint",
                coef_dict=coef_dictionary
            )

    elif str(level) == "2":
        print("[+] Level 2 Selected: Total Volume Known, Locations Unknown")

        # All parameters remain active
        for sp in sp_values:
            sp_mask = pst.parameter_data["sp"] == int(sp)
            sp_pars = pst.parameter_data.loc[sp_mask]
            par_list = list(sp_pars.index)

            print(f"[DEBUG] Level 2, sp={sp}, matched parameters={len(par_list)}")

            if len(par_list) == 0:
                raise ValueError(f"No parameters matched for stress period {sp}")

            coef_dictionary = {par: 1.0 for par in par_list}

            pst.add_pi_equation(
                par_names=par_list,
                pilbl=f"tot_q_sp{int(sp)}".lower(),
                rhs=float(total_pumping_per_sp[sp]),
                weight=100.0,
                obs_group="pumping_constraint",
                coef_dict=coef_dictionary
            )

    elif str(level) == "3":
        print("[+] Level 3 Selected: Completely Blind Inversion (Nothing Known)")
        # No prior-information equations
        pass

    else:
        raise ValueError("level must be '1', '2', or '3'")

    # -------------------------------------------------------------------------
    # Global overrides
    # -------------------------------------------------------------------------
    pst.observation_data.loc[:, "weight"] = 1.0 / 0.10  # 10 cm sigma
    print("[+] Dynamic head observation weights uniformized to: 10.0")

    adj_mask = pst.parameter_data["partrans"] != "fixed"
    pst.parameter_data.loc[adj_mask, "partrans"] = "none"
    pst.parameter_data.loc[adj_mask, "parval1"] = 0.5
    pst.parameter_data.loc[adj_mask, "parlbnd"] = 0.01
    pst.parameter_data.loc[adj_mask, "parubnd"] = 2.0

    # -------------------------------------------------------------------------
    # PEST++ configuration
    # -------------------------------------------------------------------------
    pst.pestpp_options.update({
        "ies_num_reals": NUM_REALIZATIONS,
        "ies_autogen_par_ensem": "off",
        "ies_enforce_bounds": "false"
    })

    # Helpful debug before writing
    if hasattr(pst, "prior_information") and pst.prior_information is not None:
        print("[DEBUG] Prior information preview:")
        print(pst.prior_information.head())

    final_pst_path = os.path.join(PEST_DIR, pst_name)
    pst.write(final_pst_path, version=2)
    print(f"\n[!] Success: Control file created: {os.path.basename(final_pst_path)}")

    # -------------------------------------------------------------------------
    # Start the master-worker run
    # -------------------------------------------------------------------------
    print(f"[INFO] Starting master manager run and spinning up 16 background workers...")

    pyemu.utils.start_workers(
        worker_dir=str(PEST_DIR),
        exe_rel_path=local_exe_name,
        pst_rel_path=pst_name,
        num_workers=16,
        master_dir=str(MASTER_DIR),
        worker_root=str(WORKER_ROOT),
        port=25318,
        cleanup=True
    )


if __name__ == "__main__":
    choice = input("Enter inversion level to execute (1, 2, or 3): ").strip()
    if choice in ["1", "2", "3"]:
        run_parallel_inversion(choice)