import os
from datetime import datetime
import flopy
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================
ROOT = "coloc2"
base_dir = Path(__file__).resolve().parent
output_dir = (base_dir.parent / "sims" / ROOT / "real").resolve()
model_ws = output_dir


# ==============================================================================
# HELPERS
# ==============================================================================
def log(msg, level="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "info": "[INFO]",
        "warn": "[WARN]",
        "error": "[ERROR]",
        "debug": "[DEBUG]",
    }.get(level, "[INFO]")
    print(f"{timestamp} {prefix} {msg}")


def ensure_dir():
    """Create the output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def get_cp_cells(sim_ws, model_name, margin=25):
    """Identify perimeter control points and save their locations."""
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), verbosity_level=0)
    gwf = sim.get_model(model_name)

    nrow = int(gwf.dis.nrow.array)
    ncol = int(gwf.dis.ncol.array)

    r_start, r_end = margin, (nrow - margin) - 1
    c_start, c_end = margin, (ncol - margin) - 1
    cp_cells = []

    for c in range(c_start, c_end + 1):
        cp_cells.append((r_start, c))
        cp_cells.append((r_end, c))

    for r in range(r_start + 1, r_end):
        cp_cells.append((r, c_start))
        cp_cells.append((r, c_end))

    cp_cells = list(dict.fromkeys(cp_cells))
    log(f"Found {len(cp_cells)} unique control point cells.")

    pd.DataFrame(cp_cells, columns=["r", "c"]).to_csv(
        output_dir / "cp_locs.csv", index=False
    )
    return cp_cells, gwf


def extract_master_truth(gwf, nper):
    """Extract the real well pumping schedule for later verification."""
    wel = gwf.get_package("wel")
    spd = wel.stress_period_data

    truth_rows = []
    all_locs = []

    for p in range(nper):
        period_data = spd.get_data(key=p)
        if period_data is not None:
            for entry in period_data:
                cellid = entry["cellid"]
                r, c = cellid[1], cellid[2]

                if (r, c) not in all_locs:
                    all_locs.append((r, c))

                well_id = all_locs.index((r, c))
                truth_rows.append(
                    {
                        "well_id": well_id,
                        "r": r,
                        "c": c,
                        "sp": p,
                        "q": abs(entry["q"]),
                    }
                )

    df_truth = pd.DataFrame(truth_rows)
    df_truth.to_csv(model_ws / "master_truth.csv", index=False)
    log(f"Master truth saved for {len(all_locs)} wells.")

    return df_truth, all_locs


def build_colocated_candidates(real_locs, qref_lookup, nper):
    """
    Build colocated candidate template:
    - one activity parameter per real well per stress period
    - qref is the fixed reference pumping magnitude from the synthetic truth
    """
    rows = []
    for sp in range(nper):
        for r, c in real_locs:
            qref = float(qref_lookup[(int(r), int(c), int(sp))])
            rows.append(
                {
                    "r": r,
                    "c": c,
                    "sp": sp,
                    "qref": qref,
                    "a": 1.0,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(model_ws / "grid_candidates.csv", index=False)

    # Explicit parameter order metadata for the activity parameters
    df_order = df.copy()
    df_order["parnme"] = [
        f"a_r:{int(r)}_c:{int(c)}_sp:{int(sp)}"
        for r, c, sp in zip(df_order["r"], df_order["c"], df_order["sp"])
    ]
    df_order[["parnme", "r", "c", "sp"]].to_csv(model_ws / "param_order.csv", index=False)

    log(f"Saved colocated grid candidates: {len(df)} rows.")
    log("Saved parameter order metadata to param_order.csv.")


def extract_target_values(gwf, cp_cells, nper):
    """Extract absolute hydraulic head data directly from simulation binaries."""
    head_file = Path(gwf.model_ws) / f"{gwf.name}.hds"
    hds = flopy.utils.binaryfile.HeadFile(str(head_file))

    b_target = []
    for p in range(nper):
        head_array = hds.get_data(kstpkper=(4, p))
        for r, c in cp_cells:
            absolute_head_value = head_array[0, r, c]
            b_target.append(absolute_head_value)

    hds.close()
    return b_target


def gen_pest_obs(b_target, nper, num_cp):
    """Create absolute hydraulic head observation input files."""
    obs_names = [f"hd_p{p}_cp{i}" for p in range(nper) for i in range(num_cp)]

    obs_df = pd.DataFrame(
        {
            "obsnme": obs_names,
            "obsval": b_target,
            "weight": 1.0,
            "obsgp": "head",
        }
    )

    obs_df.to_csv(model_ws / "obs.csv", index=False)
    obs_df.to_csv(model_ws / "dummy_obs.csv", index=False)
    log(f"Generated {len(obs_names)} absolute head observations.")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    ensure_dir()

    target_ws = input("enter simulation workspace path (sim_ws): ").strip()
    model_name = input("enter model name: ").strip()
    nper = int(input("enter number of stress periods (nper): "))

    cp_cells, gwf_model = get_cp_cells(target_ws, model_name, margin=25)
    df_truth, real_locs = extract_master_truth(gwf_model, nper)

    # Build qref lookup from the truth file
    qref_lookup = {}
    for row in df_truth.itertuples(index=False):
        qref_lookup[(int(row.r), int(row.c), int(row.sp))] = float(row.q)

    build_colocated_candidates(real_locs, qref_lookup, nper)
    b_target = extract_target_values(gwf_model, cp_cells, nper)
    gen_pest_obs(b_target, nper, len(cp_cells))

    log("Synthetic absolute head data generation complete! Now run build_pst.py.")