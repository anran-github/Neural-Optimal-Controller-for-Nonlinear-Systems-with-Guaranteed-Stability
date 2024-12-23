import scipy.io
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


# control number font size on axis.
# introduce latex
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 25})
plt.rcParams["font.family"] = "Times New Roman"


# Load the .mat file
# mat_LQR = scipy.io.loadmat('Drone_Experiments/drone_results/LQR1.MAT')
mat_NN = scipy.io.loadmat('MATLAB/drone_results/YALMIP.mat')
mat_NOM = scipy.io.loadmat('MATLAB/drone_results/multi_exp3.mat')
# Access the data from the loaded .mat file
# Assuming the data is stored in a variable named 'data'
# SS_LQR = mat_LQR['SS']
SS_NN = mat_NN['SS']
SS_NOM = mat_NOM['SS']
# t_LQR = mat_LQR['Time'][1:,0] - mat_LQR['Time'][0,0]
t_NN = mat_NN['Time'][1:,0] - mat_NN['Time'][0,0]
t_NOM = mat_NOM['Time'][1:,0] - mat_NOM['Time'][0,0]

# Plot the data
# t = np.linspace()
start = 2222
start_NN = 1345
start_NOM = 1365
end = 3820
end_NN = 2060
end_NOM = 2080

# exp 2
# start = 2222
# start_NN = 1345
# start_MultiAgt = 1220
# end = 3820
# end_NN = 2060
# end_MultiAgt = 2300

# X direction
fig = plt.figure(figsize=(10, 12)) 
plt.subplot(311)
# plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[0,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[0,start_NN:end_NN],label='YALMIP NN',linewidth=3)
plt.plot(t_NOM[start_NOM:end_NOM]-t_NOM[start_NOM],SS_NOM[0,start_NOM:end_NOM],label='Multi-Agent NN',linewidth=3)
plt.plot(np.linspace(0,10,10),np.zeros(10),'k--',label='Reference',linewidth=3)
plt.legend(labelspacing=0.1)
plt.xlim(0,10)
# Customize tick font properties
plt.tick_params(axis='x', colors='black', labelsize=25, width=1)  # X-axis ticks
plt.tick_params(axis='y', colors='black', labelsize=25, width=1)  # Y-axis ticks
plt.ylabel('X Position [m]', fontsize=25, color="black", fontweight="bold")
plt.xlabel('Time [s]', fontsize=25, color="black", fontweight="bold")

# plt.title('Drone Position Changes with Time.')
# Set specific intervals for y ticks
y_ticks = np.linspace(-0.6, 0.2, 5)  # Tick values at intervals of 5
plt.yticks(ticks=y_ticks)
# # Make grid lines equidistant
plt.grid(True, linestyle='--', linewidth=1, color='gray')  # Grid style
# plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio for x and y


# Y direction
plt.subplot(312)
# plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[1,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[1,start_NN:end_NN],label='YALMIP NN',linewidth=3)
plt.plot(t_NOM[start_NOM:end_NOM]-t_NOM[start_NOM],SS_NOM[1,start_NOM:end_NOM],label='Multi-Agent NN',linewidth=3)
plt.plot(np.linspace(0,10,10),np.zeros(10),'k--',label='Reference',linewidth=3)

plt.legend(labelspacing=0.1)
plt.xlim(0,10)
# Customize tick font properties
plt.tick_params(axis='x', colors='black', labelsize=25, width=1)  # X-axis ticks
plt.tick_params(axis='y', colors='black', labelsize=25, width=1)  # Y-axis ticks
plt.ylabel('Y Position [m]', fontsize=25, color="black", fontweight="bold")
plt.xlabel('Time [s]', fontsize=25, color="black", fontweight="bold")

# plt.title('Drone Position Changes with Time.')
# Set specific intervals for y ticks
y_ticks = np.linspace(-0.5, 0.1, 5)  # Tick values at intervals of 5
plt.yticks(ticks=y_ticks)
# # Make grid lines equidistant
plt.grid(True, linestyle='--', linewidth=1, color='gray')  # Grid style
# plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio for x and y




# Z direction 
plt.subplot(313)
# plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[2,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[2,start_NN:end_NN],label='YALMIP NN',linewidth=3)
plt.plot(t_NOM[start_NOM:end_NOM]-t_NOM[start_NOM],SS_NOM[2,start_NOM:end_NOM],label='Multi-Agent NN',linewidth=3)
plt.plot(np.linspace(0,10,10),1.5*np.ones(10),'k--',label='Reference',linewidth=3)

plt.legend(labelspacing=0.1)
plt.xlim(0,10)
# Customize tick font properties
plt.tick_params(axis='x', colors='black', labelsize=25, width=1)  # X-axis ticks
plt.tick_params(axis='y', colors='black', labelsize=25, width=1)  # Y-axis ticks
plt.ylabel('Z Position [m]', fontsize=25, color="black", fontweight="bold")
plt.xlabel('Time [s]', fontsize=25, color="black", fontweight="bold")

# plt.title('Drone Position Changes with Time.')
# Set specific intervals for y ticks
y_ticks = np.linspace(1, 1.6, 5) # Tick values at intervals of 5
plt.yticks(ticks=y_ticks)
# # Make grid lines equidistant
plt.grid(True, linestyle='--', linewidth=1, color='gray')  # Grid style
# plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio for x and y


plt.tight_layout()
plt.savefig('MATLAB/Drone_ma_yalmip.png',dpi=500)
plt.show()


# Analysis control effort.
# control_lqr = mat_LQR['CI']
'''
control_nn = mat_NN['CI']

for i in range(3):
    if i == 2:
        i += 1
    # print(i)
    # control_lqr = controls_lqr[i,:]
    # control_nn = controls_nn[i,:]
    # lqr_ctrl_sum = np.sum(np.abs(control_lqr[i,start:end]))
    nn_ctrl_sum = np.sum(np.abs(control_nn[i,start:end]))

    plt.subplot(211)
    # plt.plot(t_LQR[start:end]-t_LQR[start],control_lqr[i,start:end],label='LQR')
    _,x = plt.xlim()
    y,_ = plt.ylim()
    # plt.annotate(f'Integrate Value:{lqr_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
    plt.legend()
    plt.ylabel('LQR Input')
    plt.grid()
    plt.title('Control Signal Changes with time')
    plt.subplot(212)
    plt.plot(t_NN[start:end]-t_NN[start],control_nn[i,start:end],label='NN')
    _,x = plt.xlim()
    y,_ = plt.ylim()
    plt.annotate(f'Integrate Value:{nn_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
    plt.grid()
    plt.ylabel('NN Input')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()
    plt.close()
'''