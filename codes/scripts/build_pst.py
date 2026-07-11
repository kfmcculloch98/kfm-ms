import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyemu


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

SIMS_DIR = BASE_DIR / "sims"
RESULTS_DIR = BASE_DIR / "results"

PYTHON_LOC = Path(r"C:\Python\Personal\kfm-ms\.venv\Scripts\python.exe")

MASTER_TRUTH_FILE = "master_truth.csv"
GRID_CANDIDATES_FILE = "grid_candidates.csv"
PARAM_ORDER_FILE = "param_order.csv"


# =============================================================================
# HELPERS
# =============================================================================
def ensure_pest_dir(pest_dir):
    pest_dir.mkdir(parents=True, exist_ok=True)


def load_required_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return pd.read_csv(path)


def copy_if_exists(src, dst, label):
    if not src.exists():
        print(f"[WARN] {label} not found: {src}")
        return False
    shutil.copy(src, dst)
    print(f"[OK] Copied {label} -> {dst.name}")
    return True


def add_numeric_location_columns(pst):
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
        print("[OK] Parameter names parsed into r/c/sp successfully.")

    return pst


def build_parameter_ensemble(pst, nreals=250):
    par_df = pst.parameter_data.copy()
    num_pars = len(par_df)
    ensemble_matrix = np.zeros((nreals, num_pars), dtype=float)

    for idx, (_, row) in enumerate(par_df.iterrows()):
        partrans = str(row["partrans"]).strip().lower()
        if partrans == "fixed":
            ensemble_matrix[:, idx] = float(row["parval1"])
        else:
            lb = float(row["parlbnd"])
            ub = float(row["parubnd"])
            ensemble_matrix[:, idx] = np.random.uniform(lb, ub, size=nreals)

    pe = pyemu.ParameterEnsemble(
        pst=pst,
        df=pd.DataFrame(ensemble_matrix, columns=par_df.index)
    )
    return pe


def print_parameter_diagnostics(pst, title="[DIAG]"):
    par = pst.parameter_data.copy()
    cols = [c for c in ["parval1", "parlbnd", "parubnd", "partrans"] if c in par.columns]

    print(f"{title} Parameter summary:")
    print(par[cols].head(20))

    adj_mask = par["partrans"].astype(str).str.lower() != "fixed"
    if adj_mask.any():
        at_upper = np.isclose(par.loc[adj_mask, "parval1"], par.loc[adj_mask, "parubnd"]).sum()
        at_lower = np.isclose(par.loc[adj_mask, "parval1"], par.loc[adj_mask, "parlbnd"]).sum()
        print(f"{title} Adjustable parameters: {adj_mask.sum()}")
        print(f"{title} At upper bound: {at_upper}")
        print(f"{title} At lower bound: {at_lower}")
        print(f"{title} parval1 min/max: {par.loc[adj_mask, 'parval1'].min()} / {par.loc[adj_mask, 'parval1'].max()}")
    else:
        print(f"{title} No adjustable parameters found.")


def make_reference_grid(df_grid, level):
    """
    Modify the 'a' column level-by-level so the org folder differs by inversion level.

    Level 1:
        Preserve zero / on-off structure.
        Active rows keep a = 1.0.
        Inactive rows remain 0.0.

    Level 2:
        Flatten all a values to 1.0.

    Level 3:
        Same as Level 2 by default.
    """
    df_ref = df_grid.copy()

    if "a" not in df_ref.columns:
        raise KeyError("grid_candidates.csv must contain an 'a' column")

    if str(level) == "1":
        active = df_ref["qref"] > 0
        df_ref.loc[active, "a"] = 1.0
        df_ref.loc[~active, "a"] = 0.0

    elif str(level) == "2":
        df_ref["a"] = 1.0

    elif str(level) == "3":
        df_ref["a"] = 1.0

    else:
        raise ValueError("level must be '1', '2', or '3'")

    return df_ref


# =============================================================================
# MAIN BUILDER
# =============================================================================
def build_modular_pst(level, root_name):
    level = str(level).strip()
    if level not in {"1", "2", "3"}:
        raise ValueError("level must be one of '1', '2', or '3'")

    pest_dir = BASE_DIR / f"pest_level_{level}"
    ensure_pest_dir(pest_dir)

    source_model_ws = SIMS_DIR / root_name / "real"
    if not source_model_ws.exists():
        raise FileNotFoundError(f"Source model workspace not found: {source_model_ws}")

    truth_path = source_model_ws / MASTER_TRUTH_FILE
    grid_path = source_model_ws / GRID_CANDIDATES_FILE
    order_path = source_model_ws / PARAM_ORDER_FILE
    obs_path = source_model_ws / "obs.csv"

    df_truth = load_required_csv(truth_path, "master truth CSV")
    df_grid = load_required_csv(grid_path, "grid candidates CSV")
    df_order = load_required_csv(order_path, "parameter order CSV")
    _ = load_required_csv(obs_path, "observation CSV")

    print("[+] Reading truth and colocated activity basis...")
    print(f"[INFO] truth rows = {len(df_truth)}")
    print(f"[INFO] grid rows  = {len(df_grid)}")
    print(f"[INFO] order rows = {len(df_order)}")

    # -------------------------------------------------------------------------
    # Build level-specific grid_candidates.csv by changing 'a'
    # -------------------------------------------------------------------------
    df_grid_ref = make_reference_grid(df_grid, level)

    # Save level-specific file for inspection
    level_grid_name = f"grid_candidates_level_{level}.csv"
    level_grid_path = pest_dir / level_grid_name
    df_grid_ref.to_csv(level_grid_path, index=False)

    # Overwrite generic file that PstFrom will template
    shutil.copy(level_grid_path, pest_dir / GRID_CANDIDATES_FILE)

    print(f"[+] Wrote level-specific grid file: {level_grid_name}")
    print("[DEBUG] grid_candidates preview:")
    print(df_grid_ref.head(10))

    # Preserve concealed truth
    shutil.copy(truth_path, pest_dir / MASTER_TRUTH_FILE)

    # Copy supporting files
    shutil.copy(order_path, pest_dir / PARAM_ORDER_FILE)
    shutil.copy(obs_path, pest_dir / "obs.csv")
    shutil.copy(obs_path, pest_dir / "dummy_obs.csv")

    # -------------------------------------------------------------------------
    # Build PstFrom
    # -------------------------------------------------------------------------
    pf = pyemu.utils.PstFrom(
        original_d=str(source_model_ws),
        new_d=str(pest_dir),
        remove_existing=True,
        longnames=True
    )

    copy_if_exists(RESULTS_DIR / "G_real_basis.npy", pest_dir / "G_real_basis.npy", "G_real_basis.npy")
    copy_if_exists(RESULTS_DIR / "b_baseline_flat.npy", pest_dir / "b_baseline_flat.npy", "b_baseline_flat.npy")

    # Parameters and observations
    pf.add_parameters(
        GRID_CANDIDATES_FILE,
        par_type="grid",
        index_cols=["r", "c", "sp"],
        use_cols=["a"],
        transform="none"
    )

    pf.add_observations("obs.csv", index_cols=["obsnme"], use_cols=["obsval"])
    pf.add_py_function("surrogate.py", call_str="run_surrogate()", is_pre_cmd=False)

    pst = pf.build_pst()

    # Parse numeric coordinates
    pst = add_numeric_location_columns(pst)

    # -------------------------------------------------------------------------
    # Truth summaries
    # -------------------------------------------------------------------------
    sp_values = sorted(df_truth["sp"].dropna().unique().tolist())
    total_pumping_per_sp = df_truth.groupby("sp")["q"].sum().to_dict()

    print(f"[INFO] Unique real well locations: {len(df_truth[['r', 'c']].drop_duplicates())}")
    print(f"[INFO] Stress periods in truth: {sp_values}")

    # -------------------------------------------------------------------------
    # Level-specific prior-information logic
    # -------------------------------------------------------------------------
    if level == "1":
        print("[+] Level 1 Selected: Spatial Locations Known, Timing Known")

        true_locs = set(zip(df_truth["r"].astype(int), df_truth["c"].astype(int)))

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

            # pst.add_pi_equation(
            #     par_names=par_list,
            #     pilbl=f"tot_q_sp{int(sp)}".lower(),
            #     rhs=float(total_pumping_per_sp[sp]),
            #     weight=100.0,
            #     obs_group="pumping_constraint",
            #     coef_dict=coef_dictionary
            # )

    elif level == "2":
        print("[+] Level 2 Selected: Spatial Locations Known, Timing Unknown")

        for sp in sp_values:
            sp_mask = pst.parameter_data["sp"] == int(sp)
            sp_pars = pst.parameter_data.loc[sp_mask]
            par_list = list(sp_pars.index)

            print(f"[DEBUG] Level 2, sp={sp}, matched parameters={len(par_list)}")

            if len(par_list) == 0:
                raise ValueError(f"No parameters matched for stress period {sp}")

            coef_dictionary = {par: 1.0 for par in par_list}

            # pst.add_pi_equation(
            #     par_names=par_list,
            #     pilbl=f"tot_q_sp{int(sp)}".lower(),
            #     rhs=float(total_pumping_per_sp[sp]),
            #     weight=100.0,
            #     obs_group="pumping_constraint",
            #     coef_dict=coef_dictionary
            # )

    elif level == "3":
        print("[+] Level 3 Selected: Completely Blind Inversion (Nothing Known)")
        pass

    else:
        raise ValueError("level must be '1', '2', or '3'")

    # -------------------------------------------------------------------------
    # Multiplier setup
    # -------------------------------------------------------------------------
    pst.observation_data.loc[:, "weight"] = 1.0 / 0.10
    print("[+] Dynamic head observation weights uniformized to: 10.0")

    adj_mask = pst.parameter_data["partrans"].astype(str).str.lower() != "fixed"

    # Multiplier parameters centered on 1.0
    pst.parameter_data.loc[adj_mask, "partrans"] = "none"
    pst.parameter_data.loc[adj_mask, "parval1"] = 1.0
    pst.parameter_data.loc[adj_mask, "parlbnd"] = 0.01
    pst.parameter_data.loc[adj_mask, "parubnd"] = 10.0

    print_parameter_diagnostics(pst, title="[DIAG AFTER RESET]")

    # -------------------------------------------------------------------------
    # PEST++ settings
    # -------------------------------------------------------------------------
    NUM_REALIZATIONS = 250
    pst.model_command = f'"{PYTHON_LOC}" forward_run.py'
    pst.control_data.noptmax = 0

    pst.pestpp_options.update({
        "ies_num_reals": NUM_REALIZATIONS
    })

    # Remove stale / unsupported options
    pst.pestpp_options.pop("ies_autogen_par_ensem", None)
    pst.pestpp_options.pop("ies_autogen_obs_ensem", None)
    pst.pestpp_options.pop("ies_par_en_std_dev", None)

    print("[DEBUG] pestpp_options before write:")
    print(pst.pestpp_options)

    if hasattr(pst, "prior_information") and pst.prior_information is not None:
        print("[DEBUG] Prior information preview:")
        try:
            print(pst.prior_information.head())
        except Exception:
            print(pst.prior_information)

    # -------------------------------------------------------------------------
    # Write ensemble and PST
    # -------------------------------------------------------------------------
    pe = build_parameter_ensemble(pst, nreals=NUM_REALIZATIONS)
    pe_name = f"prior_ens_level_{level}.csv"
    pe_path = pest_dir / pe_name
    pe.to_csv(pe_path)
    pst.pestpp_options["ies_parameter_ensemble"] = pe_name

    final_pst_path = pest_dir / f"inversion_level_{level}.pst"
    pst.write(str(final_pst_path), version=2)

    print(f"\n[OK] Success: Control file created: {final_pst_path.name}")
    print(f"[OK] Success: Parameter ensemble created and saved to: {pe_path}")
    print("[OK] Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    choice = input("Select level (1, 2, or 3): ").strip()
    root_name = input("Enter root name (e.g. coloc): ").strip()

    build_modular_pst(choice, root_name)