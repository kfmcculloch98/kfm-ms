import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyemu

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PEST_DIR = SCRIPT_DIR.parent / "pest"
MODEL_WS = SCRIPT_DIR.parent / "sims" / "basis" / "real"
RESULTS_DIR = Path(r"c:\Python\Personal\kfm-ms\codes\results")

# Dynamically find virtual environment python executable
PYTHON_LOC = SCRIPT_DIR.parent / ".venv" / "Scripts" / "python.exe"
if not PYTHON_LOC.exists():
    PYTHON_LOC = SCRIPT_DIR.parent.parent / ".venv" / "Scripts" / "python.exe"
    if not PYTHON_LOC.exists():
        raise FileNotFoundError(f"Could not locate the virtual environment at: {PYTHON_LOC}")

def build_modular_pst(level, root_name):
    print(f"\nInitializing PEST setup via PstFrom (Grid CSV layout) in: {PEST_DIR}")

    if PEST_DIR.exists():
        shutil.rmtree(PEST_DIR, ignore_errors=True)
    PEST_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_WS.mkdir(parents=True, exist_ok=True)

    shutil.copy(RESULTS_DIR / "G_real_basis.npy", MODEL_WS / "G_real_basis.npy")
    shutil.copy(SCRIPT_DIR / "surrogate.py", MODEL_WS / "surrogate.py")

    new_style_path = RESULTS_DIR / f"{root_name}_true_operational_pumping.csv"
    old_style_path = RESULTS_DIR / f"{root_name}_optimised_pumping.csv"
    real_pumping_csv = new_style_path if new_style_path.exists() else old_style_path

    if not real_pumping_csv.exists():
        raise FileNotFoundError(f"Could not locate active pumping records: {real_pumping_csv}")

    df_pumping = pd.read_csv(real_pumping_csv)

    margin, well_buffer, nrow, ncol, nper = 20, 10, 116, 92, 52
    inner_rows = range(margin + well_buffer, (nrow - margin) - well_buffer + 1)
    inner_cols = range(margin + well_buffer, (ncol - margin) - well_buffer + 1)
    all_inner_coords = [(r, c) for r in inner_rows for c in inner_cols]

    if "r" in df_pumping.columns and "c" in df_pumping.columns:
        active_basis_coords = df_pumping[["r", "c"]].drop_duplicates().values.tolist()
    else:
        import random
        random.seed(42)
        active_basis_coords = random.sample(all_inner_coords, 20)

    print(f"Active wells detected: {len(active_basis_coords)}")

    p_rows = []
    for r, c in active_basis_coords:
        for p in range(nper):
            p_rows.append({"r": int(r), "c": int(c), "sp": int(p), "q_val": 0.0})
    df_pars_master = pd.DataFrame(p_rows)
    
    pars_master_path = MODEL_WS / "pumping_values.csv"
    df_pars_master.to_csv(pars_master_path, index=False)

    dest_obs_path = MODEL_WS / "obs.csv"
    if not dest_obs_path.exists():
        raise FileNotFoundError(f"Missing target observations at {dest_obs_path}. Run prep_data.py first!")

    true_obs_targets = pd.read_csv(dest_obs_path)
    sim_path = MODEL_WS / "sim_drawdown.txt"
    np.savetxt(sim_path, np.zeros(len(true_obs_targets)), fmt="%.6f")

    # Write structural placeholder matching PstFrom's native structure
    grid_placeholder = MODEL_WS / "p_inst0_grid.csv"
    pd.DataFrame({"parval1": np.zeros(len(p_rows))}).to_csv(grid_placeholder, index=False)

    # -------------------------------------------------------------------------
    # PstFrom Processing Staging
    # -------------------------------------------------------------------------
    pf = pyemu.utils.PstFrom(
        original_d=str(MODEL_WS),
        new_d=str(PEST_DIR),
        remove_existing=True,
        longnames=False
    )

    pf.add_parameters(
        filenames="pumping_values.csv",
        par_type="grid",
        index_cols=["r", "c", "sp"],
        use_cols=["q_val"],
        pargp="pump"
    )

    pf.add_observations(
        "obs.csv", 
        index_cols=["obsnme"], 
        use_cols=["obsval"], 
        prefix="h"
    )

    pf.add_py_function("surrogate.py", call_str="run_surrogate()", is_pre_cmd=False)

    pst = pf.build_pst()
    pst.model_command = f'"{str(PYTHON_LOC)}" forward_run.py'
    pst.control_data.noptmax = 0

    # -------------------------------------------------------------------------
    # Inversion Level Condition Assignment
    # -------------------------------------------------------------------------
    p_df = pst.parameter_data
    p_df.loc[:, "partrans"] = "fixed"
    p_df.loc[:, "parval1"] = 0.0
    p_df.loc[:, "parlbnd"] = 0.0
    p_df.loc[:, "parubnd"] = 150.0

    par_map_path = PEST_DIR / "pumping_grid.csv"
    if not par_map_path.exists():
        raise FileNotFoundError(f"Expected mapping file not found at: {par_map_path}")
        
    print(f"Loading shortname parameter mapping from: {par_map_path.name}")
    par_map_df = pd.read_csv(par_map_path)

    if level == "1":
        print("You selected Level 1 (Full Knowledge)")
        df_pumping_rates = pd.read_csv(real_pumping_csv, index_col=0)

        for well_idx, (r, c) in enumerate(active_basis_coords):
            col_key = str(well_idx) if str(well_idx) in df_pumping_rates.columns else well_idx
            if col_key not in df_pumping_rates.columns:
                raise KeyError(f"Could not find pumping column for well {well_idx}")

            pumping_history = df_pumping_rates[col_key].values
            for p in range(nper):
                match_rows = par_map_df[
                    (par_map_df["row"] == int(r)) & 
                    (par_map_df["col"] == int(c)) & 
                    (par_map_df["p"] == int(p))
                ]
                if not match_rows.empty:
                    for raw_short_name in match_rows.iloc[:, 0].values:
                        # FIX: Force .upper() to match PEST's uppercase internal tracking index!
                        short_name = str(raw_short_name).upper()
                        if short_name in p_df.index:
                            p_df.loc[short_name, "partrans"] = "none"
                            p_df.loc[short_name, "parval1"] = float(pumping_history[p])

    elif level == "2":
        print("You selected Level 2 (Known Locations)")
        for r, c in active_basis_coords:
            match_rows = par_map_df[
                (par_map_df["row"] == int(r)) & 
                (par_map_df["col"] == int(c))
            ]
            for raw_short_name in match_rows.iloc[:, 0].values:
                # FIX: Force .upper() to match PEST's uppercase internal tracking index!
                short_name = str(raw_short_name).upper()
                if short_name in p_df.index:
                    p_df.loc[short_name, "partrans"] = "none"
                    p_df.loc[short_name, "parval1"] = 10.0

    elif level == "3":
        print("You selected Level 3 (Blind)")
        p_df.loc[:, "partrans"] = "none"
        p_df.loc[:, "parval1"] = 1.0

    active_mask = p_df["partrans"] == "none"
    p_df.loc[active_mask, "parlbnd"] = 0.0
    p_df.loc[active_mask, "parubnd"] = 150.0

    pst.observation_data.loc[:, "weight"] = 1.0
    pst.observation_data.loc[:, "obgnme"] = "head"

    if "ies_autogen_obs_ensem" in pst.pestpp_options:
        del pst.pestpp_options["ies_autogen_obs_ensem"]

    final_pst_path = PEST_DIR / f"inversion_level_{level}.pst"
    pst.write(str(final_pst_path), version=2)

    print(f"\n[!] Success: Control file created: {final_pst_path.name}")
    print(f"Active Adjustable Parameters: {int(active_mask.sum())}")
    print(f"Observation count: {len(pst.observation_data)}")
    print(f"Parameter count: {len(pst.parameter_data)}")


if __name__ == "__main__":
    root = input("Enter root name for this run (e.g. realistic): ").strip()
    choice = input("Select level (1, 2, or 3): ").strip()

    if choice in ["1", "2", "3"]:
        build_modular_pst(choice, root)
    else:
        raise ValueError("Choice must be one of 1, 2, or 3")
