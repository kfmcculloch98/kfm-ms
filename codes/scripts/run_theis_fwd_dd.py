import math
import yaml
import numpy as np
import os
from scipy.special import exp1

def theis_drawdown(Q, T, S, r, t):
    if r <= 0.001 or Q <= 0: return 0.0
    u = (r**2 * S) / (4.0 * T * t)
    if u > 100: return 0.0 
    return (Q / (4.0 * math.pi * T)) * exp1(u)

def run_model():
    # 1. Load config
    if not os.path.exists("config.yml") or os.path.getsize("config.yml") == 0:
        return

    try:
        with open("config.yml", 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return

    # 2. Extract values
    T = float(cfg.get('T', 100.0))
    S = float(cfg.get('S', 0.001))
    t = float(cfg.get('t_eval', 365.0))
    xc = float(cfg.get('xc', 0.0))
    yc = float(cfg.get('yc', -50.0))
    
    # 3. Process wells
    wells = [k for k in cfg if k.startswith('well_')]
    if not wells:
        return

    total_q, weighted_x, weighted_y = 0.0, 0.0, 0.0
    for w in wells:
        try:
            # Handle nested dictionary structure from the .tpl
            q = float(cfg[w].get('Q', 0.0))
            wx = float(cfg[w].get('x', 0.0))
            wy = float(cfg[w].get('y', 0.0))
            total_q += q
            weighted_x += (wx * q)
            weighted_y += (wy * q)
        except:
            continue
        
    if total_q <= 0:
        com_x, com_y = 0.0, 0.0
    else:
        com_x = weighted_x / total_q
        com_y = weighted_y / total_q
    
    # 4. Math
    r_compliance = math.sqrt((com_x - xc)**2 + (com_y - yc)**2)
    s_comp = theis_drawdown(total_q, T, S, r_compliance, t)

    # 5. WRITE OUTPUT (Must be inside the function)
    # This format matches the 'l1 w [obs]' instruction
    with open("allobs.out", 'w') as f:
        f.write(f"res_xc {s_comp:.6f}\n")
        f.write(f"res_q {total_q:.6f}\n")

if __name__ == "__main__":
    run_model()