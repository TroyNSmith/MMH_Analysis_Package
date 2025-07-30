from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class Params(BaseModel):
    # Universal options
    num_segments: Optional[int] = None
    skip: Optional[float] = None
    coordinate_type: Optional[str] = None
    res_name: Optional[str] = None
    atoms: Optional[list] = None
    analysis_mode: Optional[str] = None
    # Radial analysis options
    radially_resolved: Optional[bool] = None
    pore_diameter: Optional[float] = None
    radius_buffer: Optional[float] = None
    num_radial_bins: Optional[int] = None
    # Specific options
    q_magnitude: Optional[float] = None


class Metadata(BaseModel):
    analysis_last_performed: str = None
    outputs: Path = None


class Results(BaseModel):
    q_magnitude: Optional[float] = None

