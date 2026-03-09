# storage_h5/schema.py
FORMAT_NAME = "milnerlab-run"
FORMAT_VERSION = 1

ROOT_CONFIGS = "/configs/analysis"
ROOT_RUNS = "/runs"

def run_root(run_id: int) -> str:
    return f"{ROOT_RUNS}/{run_id}"

def run_index(run_id: int) -> str:
    return f"{run_root(run_id)}/index"

def run_raw_ion_data(run_id: int) -> str:
    return f"{run_root(run_id)}/raw/ion_data"

def run_c2t_root(run_id: int) -> str:
    return f"{run_root(run_id)}/derived/c2t"

def run_analysis_root(run_id: int) -> str:
    return f"{run_root(run_id)}/analysis"

def config_path(config_id: str) -> str:
    return f"{ROOT_CONFIGS}/{config_id}"