# Cannon Ball Simuation v2 #
#  By Travis Fields - March 30, 2020   #

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


wd0 = genfromtxt('windfile0.csv', delimiter=',')
wd1 = genfromtxt('windfile1.csv', delimiter=',')
wd2 = genfromtxt('windfile2.csv', delimiter=',')
wd3 = genfromtxt('windfile3.csv', delimiter=',')
wd4 = genfromtxt('windfile4.csv', delimiter=',')
wd5 = genfromtxt('windfile5.csv', delimiter=',')
wd6 = genfromtxt('windfile6.csv', delimiter=',')
wd7 = genfromtxt('windfile7.csv', delimiter=',')
wd8 = genfromtxt('windfile8.csv', delimiter=',')
wd9 = genfromtxt('windfile9.csv', delimiter=',')
wd10 = genfromtxt('windfile10.csv', delimiter=',')
wd11 = genfromtxt('windfile11.csv', delimiter=',')
wd12 = genfromtxt('windfile12.csv', delimiter=',')
wd13 = genfromtxt('windfile13.csv', delimiter=',')

pred_wind_data_list = [wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7,wd8,wd9,wd10,wd11,wd12,wd13]
#pred_wind_data = genfromtxt('Cannon_Winds_Pred.csv', delimiter=',')
# Column 1 = altitude
# Column 2 = wind Vx
windfile_l =[0,1,2,3,4,5,6,7,8,9,10,11,12,13]

DegtoRad = 3.14/180 # Convert degrees to radians

R = 0.1 #Ball radius
S = 3.14*R**2/4 # Ball Wetted area
Cd = .05 # Ball drag coefficient
M = 13 # Mass of ball, kg
I = 4/3*M*R**2 # Mass moment of inertia of ball, kg-m^2
# ---------------------------- #

# Simulation Parameters #
#v = 107 # Initial velocity (muzzle velocity), m/s
#angle = 30*DegtoRad # Initial launch angle, rad
z_init = 1250 # Starting altitude, m, currently z must be > z_end
z_end = 1200 # Ending altitude, m
dt = 1 # Time step, s
# --------------------- #

# Monte Carlo Parameters #
x_dist_des = 150 # Desired downrange mean (for quantifying good/bad)
x_delta = 25  # Range +/- from desired downrange that is allowable (good)
num_sims = 100
 # Number of simulations for EACH input configuration
Vx_std = 1.0 # Set standard deviation for wind x-vel
Cd_std = 0.005 # Drag coefficient standard deviation
S_std = 0.002 # Area standard deviation
M_std = 0.5 # Mass standard deviation
#v_search = [80, 100, 400] # Paired initial velocity and angle that on average
#angle_search = [60*DegtoRad, 40*DegtoRad, 7*DegtoRad] # are close to 500m

angle_search = np.linspace(25,75,14)*DegtoRad # Launch angles to search
v_search = [500,500,500,500,500,500,500,500,500,500,500,500,500,500] # Muzzle velocities to search
#angle_search = np.linspace(5,80,10)*DegtoRad # Launch angles to search

# --------------------- #


# Altitude simulation. It starts with initial altitude and runs until the 
# next altitude is at or below the ending altitude. 
def z_sim(z_init, z_end, dt,S, Cd, M, v, angle):
    g = 9.81 # SI units
    z = np.ones(1) # Initializing altitude array, will append as sim runs
    zd = np.ones(1) # Init. z dot array
    zdd = np.ones(1) # Init. z double dot array
    t = np.ones(1) # Initializing time array, will append as sim runs
    t[0] = 0
    z[0] = z_init
    zd[0] = v*math.sin(angle) # Initial velocity from init. speed and angle
    zdd[0] = 0
	
    i = 0
    while (z[i] > z_end): # Looping until altitude is below the ground
        i = i + 1
        zdd = np.append(zdd, -g - np.sign(zd[i-1])*1/M*0.5*rho(z[i-1])*Cd*S*zd[i-1]**2)
        # z accel is weight and drag only (no other external force/propulsion)
        
        zd = np.append(zd, zd[i-1] + zdd[i]*dt)
        # z velocity is simple kinematic integration, vel = vel_prev + accel*dt
        
        z = np.append(z, z[i-1] + zd[i]*dt + 0.5*zdd[i]*dt**2)
        # altitude is simple kinematic integration
        
        t = np.append(t, t[i-1] + dt) # Simple, but sticking in here for convenience
    return z, zd, zdd, t


# X Displacement simulation. This simulation factors in the drag AND wind speed to get the
# position, velocity, and acceleration. Uses the length of the altitude sim.
# to identify when the simulation is over.
def x_sim(z, t, dt, v, angle, M, Cd, S, Vx_std, wind):
    x = np.ones(1) # Init position array
    xd = np.ones(1) # Init velocity array
    xdd = np.ones(1) # Init acceleration array
    x[0] = 0 # Setting initial conditions
    xd[0] = v*math.cos(angle)
    xdd[0] = 0
    wind_interp_x = np.ones(len(z)) # Init wind interpolation (need wind at z(k) altitude, not altitudes given in file)
    wind_interp_x[0] = 0.0
    
    for i in range(1,len(z)):
        wind_interp_x[i] = np.interp(z[i],wind[:,1], wind[:,0])+np.random.normal(0,Vx_std)
        # this linear interpolation uses the wind data (from file) and the previous altitude (z[i-1]) to estimate what the wind is at z[i-1]
        
        xdd = np.append(xdd, -np.sign(xd[i-1] - wind_interp_x[i])*1/M*0.5*rho(z[i-1])*Cd*S*(xd[i-1] - wind_interp_x[i])**2)
        # Acceleration is equal to -1/2*rho*cd*S*v^2 (drag), no external force/propulsion
        
        xd = np.append(xd, xd[i-1] + xdd[i]*dt)# + wind_interp_x[i])
        # velocity is simply kinematic integration. Xd = Xd_prev + Accel*dt + wind velocity (assuming projectile is moved by wind velocity for each time step)
        
        x = np.append(x, x[i-1] + xd[i]*dt + 0.5*xdd[i]*dt**2)
        # position is simply kinematic integration 
    return x, xd, xdd
	
# Air density is calculated based upon standard density lapse rate
def rho(alt):
    C1 = -3.9142e-14
    C2 = 3.6272e-9
    C3 = -1.1357e-4
    C4 = 1.2204
    rho_poly = C1*alt**3 + C2*alt**2 + C3*alt + C4
    return rho_poly

####################################
##### Main Simulation/Program ######
####################################


####   MONTE CARLO Simulation  #####

end_x = np.ones((num_sims,len(v_search), len(angle_search))) # Stores all landing x-distances
end_x_mean = np.zeros((len(v_search),len(angle_search))) # Stores mean of each input setup
end_x_std = np.zeros((len(v_search),len(angle_search)))  # Stores st. dev of each input setup

end_within_bnds = np.zeros((len(v_search),len(angle_search))) # Stores number of sims that meet bound criteria

Cd_all = np.ones((num_sims, len(v_search), len(angle_search))) # Stores drag coefficient for each sim
S_all  = np.ones((num_sims, len(v_search), len(angle_search))) # Stores wetted area used for each sim
M_all  = np.ones((num_sims, len(v_search), len(angle_search))) # Stores mass used for each sim

v_sim = v_search[0]
### Running the Monte Carlo Simulation ###
for i in tqdm(range(0,len(pred_wind_data_list))):
    pred_wind_data = pred_wind_data_list[i]
    for j in range(0,len(angle_search)):    # Cycle through each input setup
        # Simulating through the different initial conditions to quantify sensitivity
        angle_sim = angle_search[j]
        for k in range(0,num_sims): # tqdm just gives a progress bar, helpful when running very long simulations
            Cd_sim = Cd + np.random.normal(0,Cd_std) # Getting next Cd based upon Cd + reasonable variation
            Cd_all[k,i,j] = Cd_sim # Storing Cd so we can review later (if needed)
            S_sim = np.random.normal(S, S_std)
            S_all[k,i,j] = S_sim
            M_sim = np.random.normal(M, M_std)
            M_all[k,i,j] = M_sim
    
            z,zd, zdd, t = z_sim(z_init, z_end, dt, S_sim, Cd_sim, M_sim, v_sim, angle_sim) # Run z (altitude) simulation
            x, xd, xdd = x_sim(z, t, dt, v_sim, angle_sim, M_sim, Cd_sim, S_sim, Vx_std, pred_wind_data) # Run x (downrange) simulation
            end_x[k,i,j] = x[len(x)-1] # Storing the distance downrange (last x position) for analysis (all we care about is where it hit)
        end_x_mean[i,j] = np.average(end_x[:,i,j]) # Compute average for all sims with v[i] and angle[j] initial conditions
        end_x_std[i,j] = np.std(end_x[:,i,j]) # Compute standard deviation for all sims with v[i] and angle[j] iniital conditions
        ind_within_bnds = np.where(np.logical_and(end_x[:,i,j]>=x_dist_des-x_delta, end_x[:,i,j]<=x_dist_des+x_delta)) # Gets all indices that are within bounds
        end_within_bnds[i,j] = len(ind_within_bnds[0]) # Counts number of indices that are within bounds (gives percentage of sims that met criteria)
                                                # Stores as a tuple with numpy array inside. Had to index to first tuple to get correct length


# Initializing angle and speed variables so we can create informative plots and statistics
#angle_all = np.ones((num_sims, len(v_search))) # just using to have the angle for each simulation (same rows for angle and end point)
#Vx_all = np.ones((num_sims,len(v_search)))
angle_all = np.ones((num_sims, len(v_search))) # just using to have the angle for each simulation (same rows for angle and end point)
Vx_all = np.ones((num_sims,len(v_search)))

for j in range(0,len(angle_search)):
    angle_all[:,j] = angle_all[:,j]*angle_search[j]/DegtoRad # Used for plotting vs. initial angle



for i in range(0,len(v_search)): # Going through each initial condition and plotting/printing relavent data
    Vx_all[:,i] = Vx_all[:,i]*v_search[i] # Just creating correct length arrays for plotting vs. initial speed.    
    for j in range(0,len(angle_search)):           
        plt.figure(5)
        plt.plot(end_x[:,i,j],Vx_all[:,i], '.')
        plt.figure(6)
        plt.plot(end_x[:,i,j],angle_all[:,j], '.')
#        print "Vel: ",v_search[i], "Ang: {0:1.0f}".format(angle_search[j]/DegtoRad), ", Mean Range: {0:1.1f}".format(end_x_mean[i,j]), ", StD Range: {0:1.1f}".format(end_x_std[i,j]), ", Num in Bnds: {0:1.1f}%".format(end_within_bnds[i,j]*100/num_sims)
        
plt.figure(5)
plt.xlabel('Landing distance [m]')
plt.ylabel('Initial Velocity [m/s]')

plt.figure(6)
plt.xlabel('Landing distance [m]')
plt.ylabel('Initial Angle [deg]')


#### Creating two representative plots to examine the overall confidence in each initial condition ####

for i in range(0,len(v_search)):
    plt.figure(7)
    plt.errorbar(angle_all[0,:], end_x_mean[i,:], 2*end_x_std[i,:], marker='.', label=int(windfile_l[i]))
   
plt.figure(7)
plt.xlabel('Launch Angle, Deg')
plt.ylabel('Landing Distance')
plt.legend()

for i in range(0,len(angle_search)):
    plt.figure(8)
    plt.errorbar(Vx_all[0,:], end_x_mean[:,i], 2*end_x_std[:,i], marker='.',label=int(angle_search[i]/DegtoRad))
   
plt.figure(8)
plt.xlabel('Muzzle Velocity, m/s')
plt.ylabel('Landing Distance')
plt.legend()


### 3D Plot to look at the impact of angle and velocity on the desired impact landing percentage ###
fig = plt.figure(9)

ax = fig.add_subplot(111,projection='3d')
for i in range(0,len(windfile_l)):
    ax.plot3D(np.ones(len(angle_search))*windfile_l[i], angle_all[i,:], end_x_mean[i,:], 'x')
ax.set_xlabel('wind file number')
ax.set_ylabel('Launch Angle, deg')
ax.set_zlabel('x_mean_distance')

plt.show()

