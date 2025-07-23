import numpy as np
import mdtraj as md
from typing import Callable, Tuple


def shifted_correlation(
    function: Callable,
    Universe: md.Trajectory,
    segments: int = 1000,
    window: float = 0.9,
    skip: float = 0.0,
    average: bool = True,
    lags: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    assert window + skip < 1

    n_frames = Universe.n_frames

    max_offset = int(n_frames * window)
    skip_offset = int(n_frames * skip)

    if lags is None:
        lags = np.unique(np.logspace(0, np.log10(max_offset), num=30, dtype=int))
        lags = lags[lags < max_offset]

    max_lag = max(lags)

    latest_start = n_frames - max_lag - 1
    start_frames = np.unique(
        np.logspace(0, np.log10(latest_start), num=segments, endpoint=False, dtype=int)
    )
    start_frames = start_frames + skip_offset
    start_frames = start_frames[start_frames + max_lag < n_frames]

    print(start_frames, lags)

    for start in start_frames:
        start_indices = Universe[start].xyz
        segment_result = []

        for lag in lags:
            frame_indices = Universe[start + lag].xyz
            print(frame_indices)
            value = function(start_indices, frame_indices)
            segment_result.append(value)
