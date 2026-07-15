import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu
from pathlib import Path

# function added thru PstFrom.add_py_function()
def run_surrogate():
    print("[Surrogate] Matrix forward wrapper execution initiated...", flush=True)
    print(f"[Surrogate] CWD: {os.getcwd()}", flush=True)

    print("[Surrogate] Directory listing:", flush=True)
    for f in sorted(os.listdir(".")):
        print(f"   {f}", flush=True)

    # -------------------------------------------------------------------------
    # 1. LOAD STATIC FILES
    # -------------------------------------------------------------------------
    g_path = Path("G_real_basis.npy")
    b_path = Path("b_baseline_flat.npy")
    basis_path = Path("grid_candidates.csv")
    obs_template_path = Path("dummy_obs.csv")

    if not g_path.exists():
        raise FileNotFoundError(f"Missing required file: {g_path.resolve()}")
    if not b_path.exists():
        raise FileNotFoundError(f"Missing required file: {b_path.resolve()}")
    if not basis_path.exists():
        raise FileNotFoundError(f"Missing required file: {basis_path.resolve()}")
    if not obs_template_path.exists():
        raise FileNotFoundError(f"Missing required file: {obs_template_path.resolve()}")

    G = np.load(g_path)
    b_baseline_flat = np.load(b_path)

    print(f"[Surrogate] Loaded G with shape {G.shape}", flush=True)
    print(f"[Surrogate] Loaded b_baseline_flat with shape {b_baseline_flat.shape}", flush=True)

    # -------------------------------------------------------------------------
    # 2. LOAD LOCAL BASIS TEMPLATE
    # -------------------------------------------------------------------------
    basis_df = pd.read_csv(basis_path)
    basis_df.columns = basis_df.columns.str.lower()

    required_basis_cols = {"r", "c", "sp", "qref"}
    missing_basis = required_basis_cols - set(basis_df.columns)
    if missing_basis:
        raise ValueError(
            f"{basis_path.name} missing required columns: {missing_basis}"
        )

    basis_df["r"] = basis_df["r"].astype(int)
    basis_df["c"] = basis_df["c"].astype(int)
    basis_df["sp"] = basis_df["sp"].astype(int)
    basis_df["qref"] = basis_df["qref"].astype(float)

    # -------------------------------------------------------------------------
    # 3. FIND THE ACTIVE PARAMETER FILE
    # -------------------------------------------------------------------------
    preferred_files = [
        "parameters.csv",
        "par.csv",
        "inversion_level_1.par.csv",
        "inversion_level_2.par.csv",
        "inversion_level_3.par.csv",
        "inversion_level_1.par_data.csv",
        "inversion_level_2.par_data.csv",
        "inversion_level_3.par_data.csv",
    ]

    param_file = None
    for f in preferred_files:
        if Path(f).exists():
            param_file = Path(f)
            break

    if param_file is None:
        raise FileNotFoundError(
            "No parameter data source file found in active worker directory. "
            f"Looked for: {preferred_files}"
        )

    print(f"[Surrogate] Using parameter file: {param_file}", flush=True)

   # -------------------------------------------------------------------------
    # 4. LOAD PARAMETERS
    # -------------------------------------------------------------------------
    p_df = pd.read_csv(param_file, index_col=0)
    print(f"[Surrogate] Raw parameter dataframe shape: {p_df.shape}", flush=True)
    print(f"[Surrogate] Raw columns: {list(p_df.columns)}", flush=True)

    p_df.columns = p_df.columns.str.lower()

    # -------------------------------------------------------------------------
    # Case A: row-wise PEST-style table
    # Expected columns like:
    #   partrans, parval1, parlbnd, parubnd, pargp, r, c, sp, ...
    # -------------------------------------------------------------------------
    if {"parval1", "r", "c", "sp"}.issubset(set(p_df.columns)):
        print("[Surrogate] Detected row-wise parameter file format.", flush=True)

        p_df["parval1"] = pd.to_numeric(p_df["parval1"], errors="coerce")

        bad = p_df[p_df["parval1"].isna()]
        if len(bad) > 0:
            raise ValueError(
                "Non-numeric parval1 values found in parameter file:\n"
                f"{bad[['parval1']].head(10)}"
            )

        p_df["r"] = pd.to_numeric(p_df["r"], errors="coerce")
        p_df["c"] = pd.to_numeric(p_df["c"], errors="coerce")
        p_df["sp"] = pd.to_numeric(p_df["sp"], errors="coerce")

        parse_fail = p_df[p_df[["r", "c", "sp"]].isna().any(axis=1)]
        if len(parse_fail) > 0:
            print(f"[Surrogate] WARNING: {len(parse_fail)} parameters failed r/c/sp parsing.", flush=True)
            print(parse_fail[["parval1", "r", "c", "sp"]].head(10), flush=True)

        # activity / multiplier vector
        a = p_df["parval1"].to_numpy(dtype=float)

        basis_lookup = {
            (int(row.r), int(row.c), int(row.sp)): float(row.qref)
            for row in basis_df.itertuples(index=False)
        }

        try:
            qref = np.array(
                [
                    basis_lookup[(int(row.r), int(row.c), int(row.sp))]
                    for row in p_df.itertuples(index=False)
                ],
                dtype=float
            )
        except KeyError as e:
            raise KeyError(
                f"Could not match a parameter to grid_candidates.csv: missing key {e}"
            )

    # -------------------------------------------------------------------------
    # Case B: ensemble-style table
    # -------------------------------------------------------------------------
    else:
        print("[Surrogate] Detected ensemble-style parameter file format.", flush=True)

        if len(p_df) < 1:
            raise ValueError("Parameter file is empty.")

        # Use the first realization row
        row = p_df.iloc[0].copy()

        # Remove non-parameter label columns if present
        for label_col in ["real_name", "name", "realization"]:
            if label_col in row.index:
                row = row.drop(labels=[label_col])

        # Convert remaining entries to numeric
        numeric_row = pd.to_numeric(row, errors="coerce").dropna()

        print(f"[Surrogate] Numeric parameter columns found: {len(numeric_row)}", flush=True)
        print(f"[Surrogate] First 10 numeric column names: {list(numeric_row.index[:10])}", flush=True)

        a = numeric_row.to_numpy(dtype=float)

        if len(a) != G.shape[1]:
            raise ValueError(
                f"Activity vector length does not match G columns in ensemble mode: "
                f"len(a)={len(a)}, G.shape={G.shape}"
            )

        if len(basis_df) != len(a):
            raise ValueError(
                f"basis_df length does not match parameter vector length in ensemble mode: "
                f"len(basis_df)={len(basis_df)}, len(a)={len(a)}"
            )

        qref = basis_df["qref"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # 5. DIMENSION CHECKS
    # -------------------------------------------------------------------------
    print(f"[Surrogate] Activity vector length: {len(a)}", flush=True)
    print(f"[Surrogate] G columns: {G.shape[1]}", flush=True)

    if len(a) != G.shape[1]:
        raise ValueError(
            f"Activity vector length does not match G columns: len(a)={len(a)}, G.shape={G.shape}"
        )

    if len(qref) != len(a):
        raise ValueError(
            f"qref length does not match activity vector length: len(qref)={len(qref)}, len(a)={len(a)}"
        )

    q_effective = qref * a

    # -------------------------------------------------------------------------
    # 6. COMPUTE SURROGATE OUTPUT
    # -------------------------------------------------------------------------
    try:
        drawdown_sim = G @ q_effective
    except Exception as e:
        raise RuntimeError(f"Error during matrix multiplication G @ q_effective: {repr(e)}")

    # -------------------------------------------------------------------------
    # INTERCEPT & SLICE COUPLING: 70-WEEK SYSTEM CONSTRAINED TO ACTIVE 52 SCOPE
    # -------------------------------------------------------------------------
    if len(drawdown_sim) == 14840:
        print("[Surrogate] Intercepted 70-week simulation matrix stream. Slicing down to 52 weeks...", flush=True)
        n_cp = 212
        nper_active = 52
        
        # Unpack flat data into standard spatial matrix blocks
        drawdown_matrix = drawdown_sim.reshape((n_cp, 70))
        # Keep only the active 52 operational tracking weeks
        drawdown_active = drawdown_matrix[:, :nper_active]
        # Re-flatten back to an 11,024 vector to align with standard reference inputs
        drawdown_sim = drawdown_active.flatten()

    if drawdown_sim.shape != b_baseline_flat.shape:
        raise ValueError(
            f"Output length mismatch: len(drawdown_sim)={len(drawdown_sim)}, "
            f"len(b_baseline_flat)={len(b_baseline_flat)}"
        )

    b_sim = b_baseline_flat - drawdown_sim


    # -------------------------------------------------------------------------
    # 7. LOAD OBSERVATION TEMPLATE
    # -------------------------------------------------------------------------
    obs_df = pd.read_csv(obs_template_path)
    print(f"[Surrogate] Observation template shape: {obs_df.shape}", flush=True)

    if "obsnme" not in obs_df.columns:
        raise ValueError("dummy_obs.csv missing 'obsnme' column")

    obs_df["obsnme"] = obs_df["obsnme"].astype(str).str.lower()

    obs_df["p_num"] = pd.to_numeric(obs_df["obsnme"].str.extract(r"hd_p(\d+)")[0], errors="coerce")
    obs_df["c_num"] = pd.to_numeric(obs_df["obsnme"].str.extract(r"_cp(\d+)")[0], errors="coerce")

    obs_parse_fail = obs_df[obs_df[["p_num", "c_num"]].isna().any(axis=1)]
    if len(obs_parse_fail) > 0:
        print(f"[Surrogate] WARNING: {len(obs_parse_fail)} observation names failed parsing.", flush=True)
        print(obs_parse_fail[["obsnme", "p_num", "c_num"]].head(10), flush=True)

    if len(obs_df) != len(b_sim):
        raise ValueError(
            f"Observation template length does not match simulated head vector length: "
            f"len(obs_df)={len(obs_df)}, len(b_sim)={len(b_sim)}"
        )

    obs_df["obsval"] = b_sim
    obs_df = obs_df.drop(columns=["p_num", "c_num"])

    # -------------------------------------------------------------------------
    # 8. WRITE OUTPUT
    # -------------------------------------------------------------------------
    obs_df.to_csv("obs.csv", index=False)
    print(f"[Surrogate] Success. Generated obs.csv with {len(b_sim)} observation values.", flush=True)
    print(
        f"[Surrogate] Absolute Head summary: Max={np.max(b_sim):.4f}, "
        f"Min={np.min(b_sim):.4f}, Avg={np.mean(b_sim):.4f}",
        flush=True
    )

def main():

    try:
       os.remove(r'obs.csv')
    except Exception as e:
       print(r'error removing tmp file:obs.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    run_surrogate()

if __name__ == '__main__':
    mp.freeze_support()
    main()

