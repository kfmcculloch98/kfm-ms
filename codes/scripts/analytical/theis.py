import os
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.special import exp1  # Exponential integral E1(u) for Theis solution
from scipy.optimize import lsq_linear

# =====================================================================
# 1. PHYSICAL ENVIRONMENT & CALIBRATED PARAMETERS
# =====================================================================
root = input("Enter root name for this run: ").strip()

nper = 52       # 52 weeks in a year
perlen = 7.0    # 7 days per week
total_days = nper * perlen

# Grid domain equivalent for coordinate translation (meters) - Matched to Part 1
cell_dim, ncol, nrow = 100.0, 92, 116 
lx, ly = ncol * cell_dim, nrow * cell_dim

# Aquifer Properties - Aligned with your updated MODFLOW choices
K_hydraulic = 0.005        # Hydraulic conductivity (m/day)
aquifer_thickness = 1477.0 # Saturated thickness (top=2477 - botm=1000)
T_aquifer = K_hydraulic * aquifer_thickness # Transmissivity (m^2/day)

# Convert MODFLOW Specific Storage (Ss) to dimensionless Storativity (S)
Ss_modflow = 1e-5
S_aquifer = Ss_modflow * aquifer_thickness 

# General Head Boundary (GHB) Specifications
boundary_conductance = 0.05  # Matched to Part 1 value
beta_leakage = boundary_conductance / (boundary_conductance + (T_aquifer / cell_dim))

# Spatial Layout Constraints - Matched to Part 1
margin = 25      # minimum cell buffer from the outer grid edge
well_buffer = 5  # minimum cell buffer from the compliance perimeter

r_start, r_end = margin, (nrow - margin) - 1
c_start, c_end = margin, (ncol - margin) - 1
cp_cells = []

# Generate perimeter compliance points
for c in range(c_start, c_end + 1):
    cp_cells.extend([(r_start, c), (r_end, c)])
for r in range(r_start + 1, r_end):
    cp_cells.extend([(r, c_start), (r, c_end)])
n_cp = len(cp_cells)

# Restrict real well domain positions
inner_rows = range(r_start + well_buffer, r_end - well_buffer + 1)
inner_cols = range(c_start + well_buffer, c_end - well_buffer + 1)
inner_coords = [(r, c) for r in inner_rows for c in inner_cols]

n_real = 20
random.seed(42)
real_locs_raw = random.sample(inner_coords, n_real)

# Order-preserving spatial deduplication
phantom_locs = []
for loc in real_locs_raw:
    if loc not in phantom_locs:
        phantom_locs.append(loc)
n_phantom = len(phantom_locs)

# =====================================================================
# 2. MATCHING THEIS EQUATION MATRIX OPERATIONS (PURE SUPERPOSITION)
# =====================================================================
def get_theis_drawdown(distance, time_days, Q):
    """Computes basic analytical infinite transient drawdown."""
    if time_days <= 0 or distance <= 0:
        return 0.0
    u = (distance**2 * S_aquifer) / (4.0 * T_aquifer * time_days)
    return (Q / (4.0 * np.pi * T_aquifer)) * exp1(u)

def get_bounded_drawdown(well_rc, cp_rc, time_days, Q):
    """Calculates drawdown combining real well and analytical image wells."""
    w_r, w_c = well_rc
    cp_r, cp_c = cp_rc
    
    # Convert cell indices to coordinate dimensions (meters)
    w_y, w_x = w_r * cell_dim, w_c * cell_dim
    cp_y, cp_x = cp_r * cell_dim, cp_c * cell_dim
    
    # Distance to true well
    r_real = math.sqrt((cp_x - w_x)**2 + (cp_y - w_y)**2)
    s_real = get_theis_drawdown(r_real, time_days, Q)
    
    # Right Boundary Image Well Mirror Approximation
    boundary_x_right = (ncol - 1) * cell_dim
    image_x_right = boundary_x_right + (boundary_x_right - w_x)
    r_image_right = math.sqrt((cp_x - image_x_right)**2 + (cp_y - w_y)**2)
    s_image_right = get_theis_drawdown(r_image_right, time_days, -Q * beta_leakage)
    
    # Left Boundary Image Well Mirror Approximation (Added to match MODFLOW setup)
    boundary_x_left = 0.0
    image_x_left = boundary_x_left - w_x
    r_image_left = math.sqrt((cp_x - image_x_left)**2 + (cp_y - w_y)**2)
    s_image_left = get_theis_drawdown(r_image_left, time_days, -Q * beta_leakage)
    
    return s_real + s_image_right + s_image_left

# Assemble transient unit response tensor (R_full)
print(f"Compiling transient analytics for {n_phantom} unique wells...")
R_full = np.zeros((nper, n_cp, n_phantom))
scale_rate = 1.0

for j_phan in range(n_phantom):
    well_coords = phantom_locs[j_phan]
    for i_cp in range(n_cp):
        cp_coords = cp_cells[i_cp]
        for p in range(nper):
            elapsed_days = (p + 1) * perlen
            dd = get_bounded_drawdown(well_coords, cp_coords, elapsed_days, scale_rate)
            R_full[p, i_cp, j_phan] = dd / scale_rate

# =====================================================================
# 3. GLOBAL MATRIX ASSEMBLY & SCENARIO GENERATION
# =====================================================================
G = np.zeros((nper * n_cp, nper * n_phantom))
p_obs, p_pump = np.indices((nper, nper))
causal_mask = p_obs >= p_pump
elapsed_weeks = p_obs - p_pump
time_block_template = np.zeros((nper, nper))

for i_cp in range(n_cp):
    row_start = i_cp * nper
    row_end = row_start + nper
    for j_phan in range(n_phantom):
        col_start = j_phan * nper
        col_end = col_start + nper
        
        response_profile = R_full[:, i_cp, j_phan]
        time_block = time_block_template.copy()
        time_block[causal_mask] = response_profile[elapsed_weeks[causal_mask]]
        G[row_start:row_end, col_start:col_end] = time_block

# Calibrated baseline pumping rates to test drawdowns
rng = np.random.default_rng(seed=42)
real_wells_data = []
pumping_states = np.array([0, 1])

for i, (r, c) in enumerate(phantom_locs):
    base_rate = rng.uniform(10, 150)
    rates = base_rate * rng.choice(pumping_states, size=nper)
    real_wells_data.append({"well_id": i, "r": r, "c": c, "Q": rates})

# Construct target drawdown vector analytically using linear superposition
print("Computing reference model forward solution...")
b_target_list = []
for i_cp in range(n_cp):
    cp_coords = cp_cells[i_cp]
    for p_ob in range(nper):
        total_drawdown = 0.0
        for w in real_wells_data:
            well_coords = (w["r"], w["c"])
            for p_pu in range(p_ob + 1):
                pulse_rate = w["Q"][p_pu]
                current_elapsed_days = ((p_ob - p_pu) + 1) * perlen
                total_drawdown += get_bounded_drawdown(well_coords, cp_coords, current_elapsed_days, pulse_rate)
        b_target_list.append(total_drawdown)
b_target = np.array(b_target_list)

# =====================================================================
# 4. LINEAR OPTIMIZATION INVERSION
# =====================================================================
print(f"Solving for {nper * n_phantom} optimized pumping rates...")
res = lsq_linear(G, b_target, bounds=(0, np.inf), method="trf", max_iter=1000)

print(f"Optimization complete. Solver iterations: {res.nit}")

# Build a metadata dictionary of the solver results
optimization_stats = {
    "root": root,
    "iterations": res.nit,
    "success": str(res.success),
    "status_message": res.message,
    "cost": res.cost
}

script_dir = Path(__file__).parent
results_path = script_dir.parent / "results"
results_path.mkdir(parents=True, exist_ok=True)

# Save to a diagnostic CSV in the results folder
stats_df = pd.DataFrame([optimization_stats])
stats_df.to_csv(results_path / f"{root}_optimization_metadata.csv", index=False)
print(f"Optimization metadata successfully saved to: {results_path}")

# Save the optimized pumping rates and drawdown matches
q_flat_pest = res.x
b_match = G @ q_flat_pest

# Structural re-indexing to spreadsheet format (Rows = Time, Columns = Location)
q_opt = q_flat_pest.reshape((n_phantom, nper)).T
target_reshaped = b_target.reshape((n_cp, nper)).T
match_reshaped = b_match.reshape((n_cp, nper)).T

# Save Outputs
pd.DataFrame(target_reshaped).to_csv(results_path / f"{root}_target_drawdown.csv", index_label="SP")
pd.DataFrame(q_opt).to_csv(results_path / f"{root}_optimised_pumping.csv", index_label="SP")
pd.DataFrame(match_reshaped).to_csv(results_path / f"{root}_surrogate_match.csv", index_label="SP")

np.save(results_path / "G_real_basis.npy", G)
print(f"Data files and G matrix written out to: {results_path}")

# =====================================================================
# 5. DIAGNOSTIC SCATTER GENERATION
# =====================================================================
real_cum_flat = target_reshaped.flatten()
surrogate_cum_flat = match_reshaped.flatten()

fig1, ax1 = plt.subplots(figsize=(8, 8))
vals_all = np.concatenate([real_cum_flat, surrogate_cum_flat])
vmin, vmax = vals_all.min(), vals_all.max()
buf = (vmax - vmin) * 0.05
common_lims = (vmin - buf, vmax + buf)

ax1.set_xlim(common_lims)
ax1.set_ylim(common_lims)
ax1.plot(common_lims, common_lims, color="black", alpha=0.5, linestyle="--", lw=2, label="1:1 Line")
ax1.scatter(real_cum_flat, surrogate_cum_flat, color="#E69F00", s=15, alpha=0.7)

ax1.set_xlabel("Analytical Forward Model Drawdown (m)")
ax1.set_ylabel("Surrogate Reconstructed Drawdown (m)")
ax1.set_title("Analytical Superposition vs Inversion Verification")
ax1.legend()
plt.grid(True, alpha=0.3)
plt.show()