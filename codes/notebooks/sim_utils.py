import os
import flopy
import numpy as np

def setup_sim(name, wel_data, ws, mf6_exe, dt, nper):
    """The Factory: Builds a model instance for a worker."""
    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name=mf6_exe, sim_ws=ws)
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=[(dt, 1, 1.0)]*nper)
    
    ims = flopy.mf6.ModflowIms(sim, complexity="MODERATE")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    
    # Grid matching your original 50x50 setup
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=50, ncol=50, delr=2.0, delc=2.0, xorigin=-50.0, yorigin=-50.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=1e-3, icelltype=0) 
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, transient=True)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfwel(gwf, pname='wel', stress_period_data=wel_data)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", saverecord=[("HEAD", "ALL")])
    return sim

def worker_task(task_info):
    """The Worker: Runs one simulation and extracts perimeter drawdown."""
    j, wr, wc, cp_cells, ws, mf6_exe, dt, nper = task_info
    name = f"resp_{j}"
    
    # 1. Run the simulation
    wel_unit = {p: [((0, wr, wc), -1.0)] for p in range(nper)}
    sim = setup_sim(name, wel_unit, ws, mf6_exe, dt, nper)
    sim.write_simulation()
    success, _ = sim.run_simulation(silent=True)
    
    if not success:
        return j, None
        
    # 2. Extract results
    h_path = os.path.join(ws, f"{name}.hds")
    h_file = flopy.utils.binaryfile.HeadFile(h_path)
    h_data = h_file.get_alldata()
    
    n_cp = len(cp_cells)
    temp_results = np.zeros((nper, n_cp))
    for p in range(nper):
        for i, (cpr, cpc) in enumerate(cp_cells):
            temp_results[p, i] = 0.0 - h_data[p, 0, cpr, cpc]
            
    h_file.close()
    return j, temp_results