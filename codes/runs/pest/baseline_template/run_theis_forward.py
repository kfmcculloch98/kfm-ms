import os
import sys
import math
import yaml
import numpy as np
from scipy.special import exp1

def theis_u(r, T, S, t):
    r = float(r)
    t = float(t)
    if t <= 0:
        raise ValueError("t must be > 0 for Theis solution!")
    if r <= 0:
        raise ValueError("r must be > 0")
    return (r * r * S) / (4.0 * T * t)

def boundary_flux(Q, T, S, r, t):
    # q_r(r,t) = (Q / (2 * pi * r)) * exp(-u)
    r = float(r)
    u = theis_u(r, T, S, t)
    return (Q / (2.0 * math.pi * r)) * math.exp(-u)

def apply_param_set(cfg, selector):
    """
    selector may be:
      - None -> do nothing (use defaults in cfg)
      - int -> 0-based index into cfg['param_sets']
      - str -> name of param_set to select
    Returns updated cfg (mutates a copy).
    """
    if selector is None:
        return cfg
    ps = cfg.get('param_sets', [])
    if isinstance(selector, int):
        if selector < 0 or selector >= len(ps):
            raise IndexError("param_set index out of range")
        sel = ps[selector]
    else:
        # match by name string
        matches = [p for p in ps if p.get('name') == str(selector)]
        if not matches:
            raise KeyError(f"No param_set named '{selector}'")
        sel = matches[0]
    # apply T, S, t_eval if present
    if 'T' in sel:
        cfg['T'] = sel['T']
    if 'S' in sel:
        cfg['S'] = sel['S']
    if 't_eval' in sel:
        cfg['t_eval'] = sel['t_eval']
    return cfg

def run_theis_model(config_file, output_file, param_set_selector=None, verbose=False):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # apply chosen param_set (if any)
    cfg = apply_param_set(cfg, param_set_selector)

    T = float(cfg.get('T'))
    S = float(cfg.get('S'))
    t = float(cfg.get('t_eval', 365.0))

    results = {}
    for key in cfg:
        if key.startswith('well_'):
            Q = float(cfg[key]['Q'])
            r = float(cfg[key]['radial_dist'])
            fq = boundary_flux(Q, T, S, r, t)
            results[key] = fq
            if verbose:
                u = theis_u(r, T, S, t)
                print(f"{key}: Q={Q}, r={r}, u={u:.3e}, flux={fq:.6e}")

    with open(output_file, 'w') as f:
        for k in sorted(results.keys()):
            f.write(f"{k} {results[k]}\n")

if __name__ == "__main__":
    # CLI usage (PEST runs without args). For debugging you can call:
    # python run_theis_forward.py         # uses defaults in proj6.yml
    # python run_theis_forward.py 0       # use param_sets[0]
    # python run_theis_forward.py "Low T, Low S"
    arg = None
    if len(sys.argv) > 1:
        # try int index else string name
        try:
            arg = int(sys.argv[1])
        except ValueError:
            arg = sys.argv[1]
    # allow environment variable override: PARAM_SET
    env_arg = os.environ.get('PARAM_SET')
    if env_arg is not None:
        try:
            arg = int(env_arg)
        except ValueError:
            arg = env_arg
    run_theis_model('proj6.yml', 'allobs.out', param_set_selector=arg, verbose=False)