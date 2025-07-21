import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

Set1 = "/home/mmh/BrockportMDFiles/pore_simulations/simulations/noreservoir/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/358K/7_nvt_prod_system_short_small_timestep/analysis/xvg_files/ISF_OCT.csv"
ISF1 = np.loadtxt(Set1, delimiter=',')
t_A = ISF1[:,0]
y_A = ISF1[:,1]

Set2 = "/home/mmh/BrockportMDFiles/pore_simulations/simulations/noreservoir/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/358K/5_nvt_prod_system/analysis/xvg_files/ISF_OCT.csv"
ISF2 = np.loadtxt(Set2, delimiter=',')
t_B = ISF2[:,0]
y_B = ISF2[:,1]

start_B = t_B >= max(t_A[1:])

plt.plot(t_A,y_A,color='blue')
plt.plot(t_B[start_B],y_B[start_B],color='red')
plt.xscale('log')
plt.show()