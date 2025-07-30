from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .output import SummaryLogger, next_backup_name
from .basemodels import Results

class BaseAnalysis(ABC):
    @abstractmethod
    def calculate(self):
        raise NotImplementedError
    
    @abstractmethod
    def plot():
        raise NotImplementedError
    
    def save(self, output_directory: Path, module_name: Optional[str] = None, filename: str = "results.csv"):
        """
        Save results with backup and log metadata.

        :param output_directory: Folder to write outputs and summary log.
        :param module_name: Optional module name (used for log grouping).
        :param filename: Output CSV filename.
        """
        if not self.has_results:
            raise LookupError("Run calculate() before saving.")
        
        if not hasattr(self, "_results"):
            self._results = Results

        module_name = module_name or self.__class__.__name__.replace("Function", "")

        out_dir = Path(output_directory) / module_name
        out_dir.mkdir(parents=True, exist_ok=True)

        output_path = next_backup_name(out_dir / filename)
        self._results_df.to_csv(output_path, index=False)

        self._metadata.outputs = str(output_path)

        logger = SummaryLogger(output_directory)
        logger.log(
            module=module_name,
            params=self._params,
            metadata=self._metadata,
            results=self._results,
        )

    def should_run(self, output_directory: Path, override: bool = False) -> bool:
        if override:
            return True

        module_name = self.__class__.__name__.replace("Function", "")
        logger = SummaryLogger(output_directory)

        for entry in logger.data.get(module_name, []):
            if entry.get("params") == self._params.model_dump():
                return False

        return True