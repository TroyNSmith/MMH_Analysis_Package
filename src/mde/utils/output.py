import yaml

from typing import Dict, Any, Optional
from pathlib import Path

from .basemodels import Params, Metadata, Results


class SummaryLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path) / "Summary_log.yaml"
        self.data = self._load_log()

    def _load_log(self) -> Dict[str, Any]:
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_log(self):
        with open(self.log_path, "w") as f:
            yaml.dump(self.data, f, sort_keys=False)

    def _match_params(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return a == b

    def log(
        self,
        module: str,
        params: "Params",
        metadata: "Metadata",
        results: Optional["Results"] = None,
    ):
        results = results or Results()
        params_dict = params.model_dump()
        metadata_dict = metadata.model_dump()
        results_dict = results.model_dump(exclude_none=True)

        # Ensure outputs is a list of strings
        outputs = metadata_dict.get("outputs", [])
        if isinstance(outputs, str):
            outputs = [outputs]
        elif not isinstance(outputs, list):
            outputs = []

        metadata_dict["outputs"] = [str(o) for o in outputs]

        if module not in self.data:
            self.data[module] = []

        log_dir = self.log_path.parent
        new_outputs = metadata_dict["outputs"]

        matched = False
        for entry in self.data[module]:
            if self._match_params(entry["params"], params_dict):
                existing_outputs = entry.get("metadata", {}).get("outputs", [])
                combined = list(set(existing_outputs + new_outputs))
                valid_outputs = [o for o in combined if (log_dir / o).exists()]
                metadata_dict["outputs"] = valid_outputs

                entry["metadata"] = metadata_dict
                entry["results"] = results_dict
                matched = True
                break

        if not matched:
            self.data[module].append(
                {
                    "params": params_dict,
                    "metadata": metadata_dict,
                    "results": results_dict,
                }
            )

        # Prune other logs of these outputs
        if new_outputs:
            for mod, entries in self.data.items():
                for entry in entries:
                    if not self._match_params(entry["params"], params_dict):
                        outputs = entry.get("metadata", {}).get("outputs", [])
                        updated_outputs = [o for o in outputs if o not in new_outputs]
                        entry["metadata"]["outputs"] = updated_outputs

        self._save_log()


def next_backup_name(filepath: Path) -> Path:
    """
    Return the next available backup filename (e.g., file_1.csv) without modifying the original file.
    """
    i = 1
    while True:
        backup_path = filepath.with_name(f"{filepath.stem}_{i}{filepath.suffix}")
        if not backup_path.exists():
            return backup_path
        i += 1