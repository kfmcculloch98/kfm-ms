# run_theis_forward.py  -- outputs Theis drawdown s(r,t) per well
import math
import yaml
import sys
from pathlib import Path
from scipy.special import exp1  # E1 function

def theis_u(r, T, S, t):
    r = float(r)
    t = float(t)
    if t <= 0:
        raise ValueError("t must be > 0 for Theis solution!")
    if r <= 0:
        raise ValueError("r must be > 0")
    return (r * r * S) / (4.0 * T * t)

def theis_drawdown(Q, T, S, r, t):
    """
    Theis drawdown s(r,t) = (Q / (4 * pi * T)) * E1(u)
    where u = r^2 S / (4 T t)
    """
    u = theis_u(r, T, S, t)
    W = exp1(u)
    return (Q / (4.0 * math.pi * T)) * W

def run_theis_model(config_file="proj6.yml", output_file="allobs.out", verbose=False):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    T = float(cfg.get('T'))
    S = float(cfg.get('S'))
    t = float(cfg.get('t_eval', 365.0))

    results = {}
    for key in sorted(cfg):
        if key.startswith('well_'):
            Q = float(cfg[key]['Q'])
            r = float(cfg[key]['radial_dist'])
            s = theis_drawdown(Q, T, S, r, t)
            results[key] = s
            if verbose:
                u = theis_u(r, T, S, t)
                print(f"{key}: Q={Q}, r={r}, u={u:.3e}, s={s:.6e}")

    with open(output_file, 'w') as f:
        for k in sorted(results.keys()):
            f.write(f"{k} {results[k]}\n")

if __name__ == "__main__":
    # support optional CLI arg to select param_set (existing script supports it),
    # but default behavior uses proj6.yml in cwd.
    verbose = False
    if len(sys.argv) > 1 and sys.argv[1] in ("-v","--verbose"):
        verbose = True
    try:
        run_theis_model("proj6.yml", "allobs.out", verbose=verbose)
    except Exception as e:
        import traceback
        print("MODEL ERROR:", e)
        traceback.print_exc()
        sys.exit(1)