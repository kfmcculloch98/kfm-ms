import os
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pyemu


# configure paths
base_dir = Path(__file__).resolve().parent
output_dir = (base_dir.parent / "sims" / "realistic" / "real").resolve()
model_ws = output_dir


def ensure_dir():
    """Create the output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)

def get_cp_cells(sim_ws, model_name, margin=25):
    """Identify perimeter control points and save their locations."""
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, verbosity_level=0)
    gwf = sim.get_model(model_name)

    nrow = int(gwf.dis.nrow.array)
    ncol = int(gwf.dis.ncol.array)

    r_start, r_end = margin, (nrow - margin) - 1
    c_start, c_end = margin, (ncol - margin) - 1

    cp_cells = []

    # horizontal perimeter cells
    for c in range(c_start, c_end + 1):
        cp_cells.append((r_start, c))
        cp_cells.append((r_end, c))

    # vertical perimeter cells, excluding corners already included above
    for r in range(r_start + 1, r_end):
        cp_cells.append((r, c_start))
        cp_cells.append((r, c_end))

    cp_cells = list(dict.fromkeys(cp_cells))  # preserve order while removing duplicates
    print(f"found {len(cp_cells)} unique control point cells.")

    pd.DataFrame(cp_cells, columns=["r", "c"]).to_csv(
        output_dir / "cp_locs.csv",
        index=False,
    )

    return cp_cells, gwf

def extract_master_truth(gwf, nper):
    """Extract the Real well pumping schedule for later verification."""
    wel = gwf.get_package("wel")
    spd = wel.stress_period_data.get_data()

    truth_rows = []
    all_locs = []

    for p in range(nper):
        for entry in spd[p]:
            loc = (entry[0][1], entry[0][2])
            if loc not in all_locs:
                all_locs.append(loc)

    for p in range(nper):
        for entry in spd[p]:
            r, c = entry[0][1], entry[0][2]
            well_id = all_locs.index((r, c))
            truth_rows.append(
                {
                    "well_id": well_id,
                    "r": r,
                    "c": c,
                    "sp": p,
                    "q": abs(entry[1]),
                }
            )

    df_truth = pd.DataFrame(truth_rows)
    df_truth.to_csv(model_ws / "master_truth.csv", index=False)
    print(f"master truth saved for {len(all_locs)} wells.")


def extract_target_values(gwf, cp_cells, nper):
    """Extract heads at the control points and convert them to accurate positive drawdown."""
    head_file = Path(gwf.model_ws) / f"{gwf.name}.hds"
    hds = flopy.utils.binaryfile.HeadFile(str(head_file))
    
    # re-generate the identical initial hydraulic gradient column array (ncol=92)
    ncol = int(gwf.dis.ncol.array)
    strt_gradient = np.linspace((2377.0), (1707), ncol)

    b_target = []
    for p in range(nper):
        head_array = hds.get_data(kstpkper=(0, p))
        for r, c in cp_cells:
            # calculate the initial head at the control point location using the known gradient
            # then compute the target drawdown by subtracting the simulated head from the initial head
            h_init = strt_gradient[c]
            b_target.append(h_init - head_array[0, r, c])

    hds.close()
    return b_target


def gen_pest_obs(b_target, nper, num_cp):
    """create observation input files for PEST."""
    obs_names = [f"h_p{p}_cp{i}" for p in range(nper) for i in range(num_cp)]

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
    print(f"generated {len(obs_names)} observations.")


if __name__ == "__main__":
    ensure_dir()

    target_ws = input("enter simulation workspace path (sim_ws): ").strip()
    model_name = input("enter model name: ").strip()
    nper = int(input("enter number of stress periods (nper): "))

    # add margin argument to get_cp_cells to align with the interior compliance loop in control.py
    cp_cells, gwf_model = get_cp_cells(target_ws, model_name, margin=25)
    extract_master_truth(gwf_model, nper)
    b_target = extract_target_values(gwf_model, cp_cells, nper)
    gen_pest_obs(b_target, nper, len(cp_cells))

    print("Synthetic data generation complete! Now run control.py to build the PEST files.")