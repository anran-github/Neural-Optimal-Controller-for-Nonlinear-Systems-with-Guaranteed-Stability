'''
This file draws:

===============================================
These figures will NOT be inserted into final paper.
===============================================
Test trained NOM NN:
1. Given r=0 and one start point, plot figures of x1 and x2 change vs time.


==============================================================
Summary of Errors between xi and x_r with 10000 random points.
==============================================================
'''


import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse
from scipy.signal import cont2discrete
from control import dlqr
# from network_ori import P_Net
from network import P_Net



# ======DATA COLLECTION=======
# x,y,z directions --> 0,1,2
direction = 1
# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Using device: {device}")


# Objective function setting
parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--theta', type=float, default=0.01, help="Theta: [0.1, 0.01, 0.001]")
parser.add_argument('--resume', type=bool, default=False, help="Reference: range between [-1,1]")
args = parser.parse_args()

# DRONE system setting
alpha_x = 0.0527
alpha_y = 0.0187
alpha_z = 1.7873
alpha = [alpha_x,alpha_y,alpha_z]

beta_x = -5.4779
beta_y = -7.0608
beta_z = -1.7382
beta = [beta_x,beta_y,beta_z]
references = [0., 0., 1.5]



# % read A,B,C,D matrices:
A = np.array([[0,1], [0, -alpha[direction]]])
B=np.array([[0],[beta[direction]]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define the sample time (Ts)
Ts = 0.1  # Adjust the sample time as needed
# Discretize the system
Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), Ts, method='zoh')

Q = np.array([[100, 0],[0, 10]])
R = np.array([[0.1]])

if direction==2:
    Q = np.array([[20, 0],[0, 0.1]])

K,_,_ = dlqr(Ad,Bd,Q,R)
# Ad = torch.tensor(Ad,dtype=torch.float32)
# Bd = torch.tensor(Bd,dtype=torch.float32)
reference = references[direction]
# args constant settings:



theta=torch.tensor(args.theta).to(device)





model = P_Net(output_size=4).to(device)

# test saved model
# Create an instance of the model
# Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_theta0.01_0.21945167709165908.pth'))
weights = glob('weights/z/new2*.pth')


# ================ Weight Selection ===============================
# best weights for x and y:
# overall, little error, fast when x1<0:
weights = ['weights/x/new_model_epoch900_5934.632_u0.221.pth']
# fastest when x1>0, but a little error:
# 'weights/x/new_model_epoch450_7541.806_u0.247.pth',
# weights = [
# 'weights/x/new_model_epoch450_7541.806_u0.247.pth',
# 'weights/x/new_model_epoch900_5934.632_u0.221.pth',]

# best weights for z direction:
# weights = [
# 'weights/z/new_model_epoch225_2.276_u0.683.pth',
# 'weights/z/model50_epoch90_2.286_u0.842.pth', if you need this one, set r=0.
# ]
# =====================  Selection End =============================

# weights = [
# # 'weights/z/new2_model_epoch58_2.608_u1.479.pth',
# # 'weights/z/new2_model_epoch22_3.421_u1.522.pth',
# 'weights/z/new2_model_epoch46_3.409_u1.519.pth',
# 'weights/z/new2_model_epoch54_2.609_u1.485.pth',
# ]

weights.sort()

for weight in weights:
    print(weight)
    model.load_state_dict(torch.load(weight,map_location=device))

    # Set the model to evaluation mode (if needed)
    model.eval()





    # give a random point, plot how it goes after n iterations
    num_points = 1
    iteration = 100
    u_set = []
    x1_set = []
    x2_set = []
    delta_x_set = []
    v_dot_set = []
    arrow_trajectory = []


    # Bd=np.array([0,0.1]).reshape(2,1)
    # Ad=lambda xt: [[1, 0.1],[-0.1*9.8*np.cos(xt), 1]]


    # r = np.array([[0.]])
    r = np.array([[reference]])
    xr = np.array([[reference],[0]])
    X1,X2,R0 = np.meshgrid(np.linspace(-0.5,0,1),np.linspace(0,0,1),r)

    if direction == 2:
        # for z direction
        X1,X2,R0 = np.meshgrid(np.linspace(0.5,0,1),np.linspace(-1,0,1),r)

    input_data = np.column_stack((X1.ravel(),X2.ravel(), R0.ravel()))
    r = torch.tensor(input_data[:,-1],dtype=torch.float32)

    for t in range(iteration):

        if t == 0:
            xt = np.array([[input_data[:,0]],[input_data[:,1]]]).transpose(2,0,1)
        # p_tt = np.eye(2)

        with torch.no_grad():
            x_tem = torch.tensor(xt)
            x_tem = x_tem.type(torch.float32)
            # output order: p1, p2, p3, u
            t_input = torch.cat((x_tem,r.view(r.shape[0],1,1)),1).squeeze(-1)
            output = model(t_input.to(device))
            if not device.type == 'cuda':
                output = output.numpy()
            else:
                output = output.detach().cpu().numpy()

            # k, _, _ = dlqr(A, B, Q, R)
            # Ad= np.array([np.array([1, 0.1]),[-0.1*9.8*np.cos(xt.numpy()), 1]])
            # Ad= np.stack([xt[:,0]+0.1*xt[:,1],-0.1*9.8*np.sin(xt[:,0]) + 1*xt[:,1]]).transpose(1,0,2)

            # p_t = np.array([[output[0,0],output[0,1]],[output[0,1],output[0,2]]])
            # print(output)


            u_t = output[:,3]
            # compare with LQR
            # u_t = -K@(xt.squeeze(0)-xr)
    
            # saturate input u
            u_value = u_t[0]
            if u_value > 0.5:
                u_t = u_t*0+0.5
            elif u_value < -0.5:
                u_t = u_t*0-0.5

            # uncomment this line if you want to compare to dlqr.
            # u_t = -torch.from_numpy(k).type(torch.float32) @ xt
            xtt =  Ad @ xt + Bd @ u_t.reshape(-1,1,1)
            u_set.append(u_t[0])
            x1_set.append(xt[:,0])
            x2_set.append(xt[:,1])
            # our previous V

            # v_dot = np.sqrt(np.linalg.norm(xtt.T @ p_t @ xtt ,axis=1)) - np.sqrt(np.linalg.norm(xt.T @ p_tt @ xt,axis=1))
            # v_dot_set.append(v_dot)
            # u_set.append(u_t.item())
            ##======REMEMBER TO CHEKCK THETA HERE (0.01)======
            # arrow_trajectory.append((-0.01*np.linalg.norm(xt)))
            delta_x_set.append((xtt-xt).squeeze(-1))
            xt = xtt
            # p_tt = p_t


    # transfer to degrees
    delta_x_set = np.array(delta_x_set).transpose(1,0,2)
    u_set = np.array(u_set)
    x1_set = np.array(x1_set).transpose(1,0,2)
    x2_set = np.array(x2_set).transpose(1,0,2)
    v_dot_set = np.array(v_dot_set).reshape(-1)*180/np.pi

    # arrow_trajectory = np.array(arrow_trajectory)*180/np.pi



    # =================================================
    # Plot Results: x1-t, x2-t, u-t, Delta V - t, x1-x2
    # =================================================
    # introduce latex
    plt.rcParams['text.usetex'] = True

    # control number font size on axis.
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "Times New Roman"


    # control number font size on axis.
    plt.rcParams.clear()
    plt.rcParams.update()

    fig, ax = plt.subplots()

    for i in range(len(x1_set)):
        
        # arrow_trajectory = np.array(delta_x_set)*180/np.pi
        ax.quiver(x1_set[i,:,0], x2_set[i,:,0],delta_x_set[i,:,0], delta_x_set[i,:,1],
                color="C0", angles='xy',scale_units='xy', scale=2, width=.006,headaxislength=4)


    plt.grid(linestyle = '--')
    plt.xlabel(r'$x_1$ (deg)', fontsize=16)
    plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)


    # plt.title('input data changes with random points')
    plt.tight_layout()
    # plt.savefig('paper_figures/x1x2.png')
    plt.show()
    plt.close()




    # PHASE PROFILE
    
    t = Ts * np.arange(iteration)
    plt.figure()
    plt.subplot(311)
    plt.plot(t,x1_set[0,:,0])
    plt.grid('both')
    plt.xlabel('time [s]')
    plt.ylabel('x1')
    plt.title('State Profile')

    plt.subplot(312)
    plt.plot(t,x2_set[0,:,0])
    plt.xlabel('time [s]')
    plt.ylabel('x2')
    plt.grid('both')

    plt.subplot(313)
    plt.plot(t,u_set)
    plt.xlabel('time [s]')
    plt.ylabel('control input u')
    plt.grid('both')

    plt.show()
    plt.close()

