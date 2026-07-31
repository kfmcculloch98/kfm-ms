import numpy as np
import scipy.special as sp
import pandas as pd

def theis_step_response(Q, t0, t_array, T, S, r):
    out = []
    for t in t_array:
        if t <= t0:
            out.append(0.0)
        else:
            dt = t - t0
            u = (r**2 * S) / (4.0 * T * dt)
            out.append((Q / (4.0 * np.pi * T)) * sp.exp1(u))
    return np.array(out)

def run_theis():
    # read weekly schedule
    df_sched = pd.read_csv("schedule.csv")
    schedule = df_sched["pump_on"].values.astype(int)

    Q_on = 350.0
    T = 0.005 * 1477.0
    S = 1e-5 * 1477.0
    r = 500.0

    nper = len(schedule)
    perlen = 7.0
    t_edges = np.arange(0.0, nper * perlen + perlen, perlen)
    time_array = np.linspace(1.0, nper * perlen, nper)

    def theis_step_response(Q, t0, t_array, T, S, r):
        out = []
        for t in t_array:
            if t <= t0:
                out.append(0.0)
            else:
                dt = t - t0
                u = (r**2 * S) / (4.0 * T * dt)
                out.append((Q / (4.0 * np.pi * T)) * sp.exp1(u))
        return np.array(out)

    drawdown = np.zeros_like(time_array, dtype=float)
    for k in range(nper):
        if schedule[k] == 1:
            t_start = t_edges[k]
            t_end = t_edges[k + 1]
            drawdown += theis_step_response(Q_on, t_start, time_array, T, S, r)
            drawdown -= theis_step_response(Q_on, t_end, time_array, T, S, r)

    with open("drawdown.csv", "w") as f:
        for val in drawdown:
            f.write(f"{val}\n")

if __name__ == "__main__":
    run_theis()
