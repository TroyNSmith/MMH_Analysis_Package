import mdevaluate as mde
import numpy as np
import matplotlib.pyplot as plt
import helpers

testDir ="/media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/328K/5_nvt_prod_system"
rOuter = 1.2
rInner = 0.5
timestep = 0.002

mdeTraj = mde.open(testDir, "run.tpr", "out/traj.xtc", nojump=True)

octTraj = mdeTraj.subset(residue_name='OCT')

def decay_func(r):
    # Scales from 1 at r_outer to 0 at r_inner
    return (r - rInner) / (rOuter - rInner)

n_atoms = len(octTraj.atoms)
tracking = np.zeros(n_atoms, dtype=bool)
score_list = [[] for _ in range(n_atoms)]

for frame in octTraj.frames:
    octPos = frame.positions - frame._unitcell[:3] / 2
    octRad = np.linalg.norm(octPos[:,:2], axis=1)

    for i, r in enumerate(octRad):
        if not tracking[i]:
            if r > rOuter:
                tracking[i] = True
                score_list[i].append([decay_func(r)])
        else:
            if r > rInner:
                score_list[i][-1].append(decay_func(r))
            else:
                tracking[i] = False

# Flatten all event scores across all molecules
all_scores = [event for molecule_scores in score_list for event in molecule_scores]

# Optionally get lengths (residence durations)
residence_times = [len(event) * timestep for event in all_scores]

# Mean trace over time (if you want a correlation-style decay)
max_len = max(len(s) for s in all_scores)
decay_matrix = np.zeros((len(all_scores), max_len))

for i, event in enumerate(all_scores):
    decay_matrix[i, :len(event)] = event

mean_decay = decay_matrix.mean(axis=0)

plt.plot(mean_decay)
plt.xlabel("Time steps")
plt.ylabel("Mean stickiness score")
plt.title("Decay of wall-sticking events over time")
plt.show()