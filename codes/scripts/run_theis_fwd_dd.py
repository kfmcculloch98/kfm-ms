import math
import yaml
import numpy as np
from scipy.special import exp1

def theis_drawdown(Q, T, S, r, t):
    if r <= 0: return 0.0  # avoid log/zero distance errors
    u = (r**2 * S) / (4.0 * T * t)
    return (Q / (4.0 * math.pi * T)) * exp1(u)

def run_model():
    # 1. load config (written by PEST using the .tpl file)
    with open("config.yml", 'r') as f:
        cfg = yaml.safe_load(f)

    T, S, t = float(cfg['T']), float(cfg['S']), float(cfg['t_eval'])
    xc, yc = float(cfg['xc']), float(cfg['yc'])
    
    # 2. extract wells and calculate Center of Mass (CoM)
    wells = [k for k in cfg if k.startswith('well_')]
    total_q, weighted_x, weighted_y = 0.0, 0.0, 0.0
    
    for w in wells:
        q = float(cfg[w]['Q'])
        wx, wy = float(cfg[w]['x']), float(cfg[w]['y'])
        total_q += q
        weighted_x += (wx * q)
        weighted_y += (wy * q)
        
    com_x = weighted_x / total_q
    com_y = weighted_y / total_q
    
    # 3. calculate distances from CoM to targets
    r_compliance = math.sqrt((com_x - xc)**2 + (com_y - yc)**2)
    r_pit = math.sqrt(com_x**2 + com_y**2)
    
    # 4. calculate drawdown at targets based on CoM pumping
    s_comp = theis_drawdown(total_q, T, S, r_compliance, t)
    s_pit = theis_drawdown(total_q, T, S, r_pit, t)

    # 5. write output (read by PEST using the .ins file)
    with open("allobs.out", 'w') as f:
        f.write(f"s_compliance {s_comp:.6f}\n")
        f.write(f"s_pit {s_pit:.6f}\n")
        f.write(f"total_q {total_q:.6f}\n")

if __name__ == "__main__":
    run_model()
