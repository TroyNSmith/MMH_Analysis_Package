import mdtraj as mdt
import numpy as np
import pandas as pd


def gyration_tensor(mdt_coords: mdt.Trajectory):
    gyr_tensors = mdt.compute_gyration_tensor(mdt_coords)
    eigenvalues = np.linalg.eigvalsh(gyr_tensors)
    times = mdt_coords.time

    column_labels = ["t / ps", "n = 1", "n = 2", "n = 3"]

    df = pd.DataFrame(np.column_stack([times, eigenvalues]), columns=column_labels)

    return df


def radius_of_gyration(mdt_coords: mdt.Trajectory):
    rg = mdt.compute_rg(mdt_coords)
    times = mdt_coords.time

    column_labels = ["t / ps", "Rg(t)"]

    df = pd.DataFrame(np.column_stack([times, rg]), columns=column_labels)

    return df
