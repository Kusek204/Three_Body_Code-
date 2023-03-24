#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:33:27 2022

@author: riebelli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 00:26:37 2022

@author: riebelli
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

# input: 
#       - initial position and velocity,
#       - initial time, final time, dt = time step
#       - params
# return:
#       - sol_time = array of time at which the trajectory is calculated
#       - sol_state = array of position & velocity of all three masses
def integrate_euler( x0 , t0 , tf , dt , ode, params ):
    # Number of points in the dense output :
    npts = int(np.floor ((tf - t0) / dt)) + 1
    # Initial state:
    x = x0
    # Vector of times:
    sol_time = np.linspace(t0 , t0 + dt * (npts - 1), npts)
    # Store solution :
    sol_state = [x] # array of states for each discrete time
    # Main loop:
    for t in sol_time [1:]:
        # Right -hand side of the ode:
        dxdt = ode (x, t, params)
        # Advance step:
        x = x + dxdt * dt
        # Store solution :
        sol_state = np.concatenate (( sol_state , [x]))
    # Output :
    return sol_time , sol_state

#%%

def integrate_runge_kutta( x0 , t0 , tf , dt , ode, params ):
    # Allocate dense output :
    npts = int(np.floor ((tf - t0) / dt)) + 1
    # Initial state:
    x = x0
    # Vector of times:
    sol_time = np.linspace(t0 , t0 + dt * (npts - 1), npts)
    # Allocate and store initial steps:
    sol_state = np.zeros ((npts , len(x0)))
    sol_state [0,:] = x
    # Launch integration :
    for count , t in enumerate (sol_time [1:], 1):
        # Evaluate coefficients :
        k1 = ode(x, t, params)
        k2 = ode(x + 0.5 * dt * k1 , t + 0.5 * dt , params)
        k3 = ode(x + 0.5 * dt * k2 , t + 0.5 * dt , params)
        k4 = ode(x + dt * k3 , t + dt , params)
        # Advance the state :
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        # Store step:
        sol_state[count ,:] = x
    # Output :
    return sol_time , sol_state

#%%

def ode_three_body_first_order ( x, t, params ):
    # Retrieve parameters :
    G = params['G']
    masses = params['masses ']
    # Retrieve position vectors :
    R1 = x[0:3] # position vector for m1
    R2 = x[3:6] # position vector for m2
    R3 = -(masses[0]*R1+masses[1]*R2)/masses[2]
    # Compute relative separation :
    r21 = np.linalg.norm(R2 - R1)**3
    r12 = r21
    r31 = np.linalg.norm(R3 - R1)**3
    r13 = r31
    r32 = np.linalg.norm(R3 - R2)**3
    r23 = r32
    # Differential equations :
    # - Initialize velocity and acceleration:
    dxdt = np.zeros(len(x))
    # - Velocity :
    dxdt [0:3] = x[6:9]  # velocity for m1
    dxdt [3:6] = x[9:12] # velocity for m2
    # - Acceleration :
    dxdt [6:9] = -G * masses[1] * (R1 - R2) / r12 - G * masses[2] * (R1 - R3) / r13
    dxdt [9:12]= -G * masses[0] * (R2 - R1) / r21 - G * masses[2] * (R2 - R3) / r23
    return dxdt

#%%

def initialize_problem ():
    # Number of bodies :
    nbodies = 3
    # The initial conditions :
    au = 1.496e11
    sec = 24*3600
    R1 = np.array ([0.00450250878464055477, 0.00076707642709100705, 0.00026605791776697764])*au #sun
    V1 = np.array ([-0.00000035174953607552 , 0.00000517762640983341 , 0.00000222910217891203])*au/sec 
    R3 = np.array ([-5.37970676855393644523 , -0.83048132656339789295 , -0.22482887442656542236])*au #Jupitar
    V3 = np.array ([0.00109201259423733748 , -0.00651811661280738459, -0.00282078276229867897])*au/sec
    R2 = np.array ([0.12051741410138465477  , -0.92583847476914859295, -0.40154022645315222236])*au #EM Bary
    V2 = np.array ([ 0.01681126830978379448 , 0.00174830923073434441 , 0.00075820289738312913])*au/sec
    # Mass of the bodies :
    masses = [1.989e30, 5.972e24, 1.89813e27]
    # Calculate center of mass position and velocity
    RCM = (masses[0]*R1+masses[1]*R2+masses[2]*R3)/(masses[0]+masses[1]+masses[2])
    VCM = (masses[0]*V1+masses[1]*V2+masses[2]*V3)/(masses[0]+masses[1]+masses[2])
    # Shift to CM
    R1 = R1 - RCM 
    R2 = R2 - RCM
    R3 = R3 - RCM
    V1 = V1 - VCM
    V2 = V2 - VCM
    V3 = V3 - VCM
    # Store in state vector :
    x = np.concatenate ((R1 , R2 , V1 , V2)) # 12 element array (r, v)
    # Gravitational constant :
    G = 6.67e-11
    # Final time:
    t0 = 0.0
    tf = 3600*24*365
    # Time step:
    dt = 2*365
    # Tolerances :
    atol = x * 0.0 + 1e-8
    rtol = atol
    # Create parameters structure :
    params = dict()
    params['G'] = G
    params['masses '] = masses
    params['nbodies '] = nbodies
    # Output :
    return x, t0 , tf , dt , params

#%%

# Define function to integrate :
ode = ode_three_body_first_order
# The initial conditions and parameters :
x, t0 , tf , dt , params = initialize_problem ()
# Integrate orbits :
sol_time , sol_state = integrate_runge_kutta(x, t0 , tf , dt, ode , params)

#%%

def plot_trajectory( sol_state , params ):
    fig = plt.figure ()
    ax = fig.add_subplot (111)
    # Plot trajectory and initial positions with the same color :
    for ibody in range (0, params['nbodies ']):
        traj = ax.plot(sol_state [:, ibody * 3], sol_state [:,1 + ibody *3])
        ax.plot(sol_state[0, ibody *3], sol_state[0, 1+ ibody *3], "o" ,\
        color=traj[0]. get_color ())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    
#%%

def radius(x , i):
    return np.linalg.norm(x[(i-1)*3:(i-1)*3+3])

#%%

plot_trajectory(sol_state , params)


#%%
x0 = sol_state[0]
r1s = np.array(list(map(lambda x : radius(x,2) , sol_state)))
print(r1s)
plt.plot(sol_time,r1s/r1s[0])