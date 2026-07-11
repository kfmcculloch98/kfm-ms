import os
import shutil
import pyemu

PEST_DIR = r"C:\Python\Personal\kfm-ms\codes\pest"
IES_EXE = r"C:\Python\Personal\kfm-ms\codes\binaries\PESTPP\windows\pestpp-ies.exe"
WORKER_ROOT = r"C:\Python\Personal\kfm-ms\codes\pest_workers"

def run_parallel_inversion(level):
    pst_name = f"inversion_level_{level}.pst"
    print(f"\nInitializing parallel PESTPP-IES run for Level {level}...")

    if os.path.exists(WORKER_ROOT):
        try: shutil.rmtree(WORKER_ROOT)
        except OSError: os.system(f'rmdir /s /q "{WORKER_ROOT}"')
            
    os.makedirs(WORKER_ROOT, exist_ok=True)
    os.chdir(PEST_DIR)
    
    pst = pyemu.Pst(pst_name)
    pst.control_data.noptmax = 2
    
    if "ies_autogen_par_ensem" in pst.pestpp_options: del pst.pestpp_options["ies_autogen_par_ensem"]
    if "ies_autogen_obs_ensem" in pst.pestpp_options: del pst.pestpp_options["ies_autogen_obs_ensem"]
    if "ies_autogen_par_ensem" in pst.control_data.formatted_values: del pst.control_data.formatted_values["ies_autogen_par_ensem"]
        
    pe_filename = f"inversion_level_{level}.par.csv"
    print(f"Drawing parameter ensemble realizations via PyEMU bounds...")
    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst=pst, num_reals=250)
    pe.to_csv(pe_filename)
    
    pst.pestpp_options["ies_parameter_ensemble"] = pe_filename
    raw_cmd = pst.model_command if isinstance(pst.model_command, list) else pst.model_command
    pst.model_command = [raw_cmd.replace('"', '')]
    pst.write(pst_name, version=2)

    pyemu.utils.start_workers(
        worker_dir=PEST_DIR, exe_rel_path=IES_EXE, pst_rel_path=pst_name,
        num_workers=16, master_dir="master_run", worker_root=WORKER_ROOT, port=25318
    )
    
if __name__ == "__main__":
    choice = input("Enter inversion level to execute (1, 2, or 3): ").strip()
    if choice in ["1", "2", "3"]: run_parallel_inversion(choice)
