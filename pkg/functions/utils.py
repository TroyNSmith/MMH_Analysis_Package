import yaml

from datetime import datetime
from pathlib import Path

def log_analysis_yaml(log_path: Path, analysis_name: str, parameters: dict):
    """Logs metadata for each analysis step into a YAML file.

    Args:
        log_path (Path): Path to the YAML log file.
        analysis_name (str): Name of the analysis (e.g., 'msd_all').
        parameters (dict): Parameters used in the run.
    """
    log_data = {}

    if log_path.exists():
        with open(log_path, "r") as f:
            log_data = yaml.safe_load(f) or {}

    if analysis_name not in log_data:
        log_data[analysis_name] = []

    log_data[analysis_name].append({
        "Last analyzed": datetime.now().isoformat(),
        "Parameters": parameters,
    })

    with open(log_path, "w") as f:
        yaml.safe_dump(log_data, f, sort_keys=False)