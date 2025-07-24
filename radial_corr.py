import mdevaluate as mde
import numpy as np
import matplotlib as plt

testDir ="/media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/328K/5_nvt_prod_system"
rOuter = 1.2
rInner = 0.5

mdeTraj = mde.open(testDir, "run.tpr", "out/traj.xtc", nojump=True)

octTraj = mdeTraj.subset(residue_name='OCT')

n_atoms = len(octTraj.atoms)
tracking = np.zeros(n_atoms, dtype=bool)
score_list = [[] for _ in range(n_atoms)]

for frame in octTraj.frames:
    octPos = frame.positions - frame._unitcell[:3] / 2
    octRad = np.linalg.norm(octPos[:,:2], axis=1)

    for i, r in enumerate(octRad):
        if not tracking[i]:
            if r > rOuter: