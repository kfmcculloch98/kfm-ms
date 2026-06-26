import os
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent
output_dir = (base_dir.parent / "sims" / "basis" / "real").resolve()

def generate_analytical_obs(root_name):
    results_dir = base_dir.parent / "results"
    target_csv = results_dir / f"{root_name}_target_drawdown.csv"
    
    if not target_csv.exists():
        raise FileNotFoundError(f"Missing target file: {target_csv}. Please run theis.py first!")
        
    df_target_raw = pd.read_csv(target_csv, index_col=0)
    
    # In target_drawdown.csv: Rows = Stress Periods (Time), Cols = Compliance Points (Space)
    nper = df_target_raw.shape[0]  # Number of rows (52 weeks)
    num_cp = df_target_raw.shape[1] # Number of columns (212 points)

    b_target, obs_names = [], []
    
    # =====================================================================
    # CRITICAL ALIGNMENT FIX: SPACE-MAJOR, TIME-MINOR EXTRACTION
    # =====================================================================
    # To match Matrix G's row mapping hierarchy perfectly, Space must be 
    # the outer loop, and Time must be the inner loop. We explicitly use 
    # .iat[p, i] where p is row (Time) and i is column (Space).
    for i in range(num_cp):
        for p in range(nper):
            obs_names.append(f"h_p{p}_cp{i}")
            # df_target_raw index p is the row (Stress Period), column i is the CP
            b_target.append(df_target_raw.iat[p, i])

    obs_df = pd.DataFrame({
        "obsnme": obs_names, 
        "obsval": b_target, 
        "weight": 1.0, 
        "obsgp": "head"
    })
    
    obs_df.to_csv(output_dir / "obs.csv", index=False)
    print(f"Compiled {len(obs_names)} target observations safely aligned in Space-Major format.")

if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)
    root_name = input("Enter root name used in theis.py: ").strip()
    generate_analytical_obs(root_name)