import yaml

from datetime import datetime
from pathlib import Path

def log_analysis_yaml(log_path: Path, analysis_name: str, file_path: str, parameters: dict):
    """
    Logs the result of an analysis to a YAML file, overwriting previous entries with the same file path.

    :param log_path: Path to the YAML log file.
    :param analysis_name: Name of the analysis (e.g., 'msd_all').
    :param file_path: Path to the analysis output file.
    :param parameters: Parameters used in the run.
    """
    log_data = {}

    if log_path.exists():
        with open(log_path, "r") as f:
            log_data = yaml.safe_load(f) or {}

    if analysis_name not in log_data:
        log_data[analysis_name] = []

    # Remove any existing entries with the same file path
    log_data[analysis_name] = [
        entry for entry in log_data[analysis_name]
        if entry.get("Saved to") != str(file_path)
    ]

    # Add new (latest) entry
    log_data[analysis_name].append({
        "Last analyzed": datetime.now(),
        "Saved to": str(file_path),
        "Parameters": parameters,
    })

    # Save updated log
    with open(log_path, "w") as f:
        yaml.safe_dump(log_data, f, sort_keys=False)