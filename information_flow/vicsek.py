# developing...
# starting time: 2022/11/7
# last time: 2024/
# ending time: ~
# python: 3.6

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time
import os
import h5py
import progressbar

def nonlinear(x:np.ndarray):

    # tanh 
    # nonlinear_x = (2/(1+np.exp(-2*x))-1)*np.pi 
    # nonlinear_x=np.tanh(x)*np.pi #
    # nonlinear_x=np.tanh(x*2/np.pi)*np.pi #
    
    # singmoid 
    # nonlinear_x = (1/(1+np.exp(-x))-1/2)*2*np.pi
    
    # linear
    nonlinear_x = x
    return nonlinear_x



def alignment(number_of_particles:int, theta:np.ndarray, WxA:np.ndarray):
        '''
        Compute the alignment. Weighted average the velocity of adjacent units.
        
        Parameters:
        ---

        WxA: interaction matrix .* adjacent matrix

        Return:
        ---
        The result direction of alignment, whose size is (n x 2) . Each row of the matrix, (x,y), is unitized. 
        '''
        n = number_of_particles

        #dtattheta \in [-2pi,2pi]
        detatheta = -theta.reshape((n,1))+theta.reshape((1,n)) # i, j is \theta j - theta i
        
        #dtattheta \in [-pi,pi]
        detatheta = detatheta*((detatheta<=np.pi)*(detatheta>-np.pi)) + (((detatheta-2*np.pi)*(detatheta>np.pi))) + (((detatheta+2*np.pi)*(detatheta<=-np.pi)))#也可以用取余来操作

        thetaA = WxA*detatheta 
        total_weight = np.sum(WxA,axis=1).reshape((n,1))
        thetaA = np.divide(thetaA, total_weight, where = total_weight>0) #\inc\theta_ij
        nonlinearthetaA = nonlinear(thetaA)
        thetaASum = np.sum(nonlinearthetaA,axis=1) #\inc\theta_i
        
        #influence
        theta_influence = nonlinearthetaA
        
        return thetaASum, theta_influence

def update(position:np.ndarray, theta:np.ndarray, velocity:np.ndarray, l:float, n:int, r:float, Wx:np.ndarray, eta:float, now:int, speed:float,time_resolution:float):
        '''
        # Main logic
        
        update the position.        
        
        Parameters:

        '''
        dx = np.subtract.outer(position[:, 0], position[:, 0])
        dy = np.subtract.outer(position[:, 1], position[:, 1]) 
        ## periodic boundary                                         
        dx = (abs(dx)>(l/2))*(l-abs(dx))+(abs(dx)<=(l/2))*abs(dx)
        dy = (abs(dy)>(l/2))*(l-abs(dy))+(abs(dy)<=(l/2))*abs(dy)
        distance = np.hypot(dx, dy)

        
        Ax = (distance >= 0) * (distance <= r) # >=0是包括自己 
        di = np.maximum(Ax.sum(axis=1), 1) #.reshape(self.n,1)
        Dx = np.diag(di)
        Lx = Dx-Ax
        Id = np.identity(n)
       
       
        WxA = Ax * Wx
        # noise
        rng = np.random.default_rng()

        noises = rng.random((n,1))*eta - eta/2 #[-eta/2, eta/2]

        # calculate
        
        
        #angle

        # # different interaction modes
        # #A    
        dTheta, theta_influence  = alignment(number_of_particles=n, theta=theta,  WxA=WxA)#alignment                            #alignment
        theta = theta + dTheta.reshape((n,1))

        #A'
        # if now%2==0:
        #     rng = np.random.default_rng()
        #     theta = rng.random((n,1))*2*np.pi - np.pi/2
        #     dTheta = alignment(number_of_particles=n, theta=theta,  WxA=WxA) #alignment
        #     theta = theta + dTheta.reshape((n,1))
        # else:
        #     dTheta = alignment(number_of_particles=n, theta=theta,  WxA=WxA)#alignment
        #     theta = theta + dTheta.reshape((n,1))

        
        # add noises
        theta = theta + noises
        # theta[:self.indexOfLeader+1] += Noises[:self.indexOfLeader+1]
        # theta[:self.indexOfLeader+1] -= Noises[:self.indexOfLeader+1] # exclude the noise of leader

        # leader control
        # theta= self.update_Leader_Control()
        # interval range of angle->[0,2*pi)
        theta = np.mod(theta, 2 * np.pi)
        
        # speed remains unchanged
        # velocity
        velocity = np.hstack((speed*np.cos(theta),speed*np.sin(theta)))
        theta = np.arctan2(velocity[:,1].reshape(n,1),velocity[:,0].reshape(n,1))###############3
        # interval range of angle->[0,2*pi)
        theta = np.mod(theta, 2 * np.pi)########################

        # position
        position  = position + velocity * time_resolution # 1 means time
        position = np.mod(position, l)
        
        return position, theta, Ax, noises, theta_influence

frameFlag = True


def simulation(i):
    ## save the data
    # print("\n",i,"\n")
    global Data_theta, Data_velocity, Data_position, Data_noises, Data_adjacent_matrix, Data_angle_information
    global theta, velocity, position, pbar
    global frameFlag

    if frameFlag == True:
        frameFlag = False
        return
    Data_theta[:,i]=theta.reshape((number_of_particles,))
    Data_velocity[:,:,i]=velocity.reshape((number_of_particles, 2,))
    Data_position[:,:,i]=position.reshape((number_of_particles, 2,))
    

    ## draw the figure
    
    
    # print(x_lable_text)
    plt.cla()
    plt.xlabel(xlabel="$t_n = $"+str(i),size = 15) # 真实与显示可能略微有些误差

    ###draw the arrow
    plt.title(title_name)
    plt.xlim((0, size_of_arena))
    plt.ylim((0, size_of_arena))
    if number_of_influencer>=1:
        plt.quiver(position[0:number_of_influencer,0],position[0:number_of_influencer,1],np.cos(theta[0:number_of_influencer]),np.sin(theta[0:number_of_influencer]),color = 'r',width = 0.005)
    if number_of_particles > number_of_influencer:
        plt.quiver(position[number_of_influencer:,0],position[number_of_influencer:,1],np.cos(theta[number_of_influencer:]),np.sin(theta[number_of_influencer:]),width = 0.005)
    
    ### serial number
    for j in range(np.shape(position)[0]):
        plt.text(position[j,0]-0.1, position[j,1]+0.1, str(j), color="brown")

    ### draw the trajectory
    trajL = 15 # trajectory length
    scatterSize = np.ones((number_of_particles,trajL))*0.5 # trajectory size
    if i > trajL:
        plt.scatter(Data_position[:,0,i-trajL:i], Data_position[:,1,i-trajL:i], marker='.', s=scatterSize[:,:], color='k')#画轨迹点
    else:
        plt.scatter(Data_position[:,0,:i], Data_position[:,1,:i], marker='.', s=scatterSize[:,:i], color='k')#画轨迹点

    ## update
    position, theta, Ax, noises, theta_influence= update(position=position,theta=theta,velocity=velocity,
                                    l=size_of_arena,n=number_of_particles,r=sense_radius,Wx=Wx,eta=eta,
                                    now=i, speed=speed, time_resolution=time_resolution)
    
    Data_adjacent_matrix[:,:,i] = Ax.reshape((number_of_particles, number_of_particles))
    Data_angle_information[:,:,i]=theta_influence[:,:]
    if i==0:
        Data_noises[:,i]=np.zeros(number_of_particles,)
    else:
        Data_noises[:,i]=noises.reshape((number_of_particles,))
    pbar.update(i+1)




if __name__=="__main__":

    #%%
    # Initialize parameters of Vicsek model:
    number_of_particles = 50  # the number of particles
    number_of_influencer = 1 # number of influencer
    size_of_arena = 10 # he linear size of the squrae shape cell where simulations are carried out
    speed = 2 # speed of particles
    sense_radius = 3 # the radius that one particles can sense
    eta = 0.2*np.pi # eta, the noise strength
    rng = np.random.default_rng() #
    # init Position P
    position = rng.random((number_of_particles,2))*size_of_arena # random()-> float interval [0,1]
    # init orientation   Theta        0~2*pi 
    theta = rng.random((number_of_particles, 1))*2*np.pi 
    # init Velocity  Ve    
    velocity = np.hstack((speed*np.cos(theta),speed*np.sin(theta)))
    
    # init interaction strength
    Wx = np.ones((number_of_particles,number_of_particles))
    wLF = [5.0, 1] # for two leaders
    Wx[0:number_of_influencer,number_of_influencer:] = 0 # followers' effect on leader
    Wx[0:number_of_influencer,0:number_of_influencer] = 0 #  leader on leader
    Wx[number_of_influencer:,number_of_influencer:] =0 # ; follower on follower
    followerIndex = np.arange(number_of_influencer,number_of_particles)
    for i in followerIndex: #follower on itself
        Wx[i,i]=1
    for i in range(number_of_influencer): #leader on itself
        Wx[number_of_influencer:,i] = wLF[i]  # leader's effect on followers # leader 之间没有相互影响
        Wx[i,i]=wLF[i]
    # print("Wx\n", self.Wx[:5,:5])

    #%%
    # Initialize data and simulation
    number_of_steps = 10 # 2 power of 17
    fps = 20 
    time_resolution = 1/fps 
    ## Data initialization
    Data_theta = np.zeros((number_of_particles, number_of_steps))
    Data_velocity = np.zeros((number_of_particles,2,number_of_steps))
    Data_adjacent_matrix = np.zeros(((number_of_particles,number_of_particles,number_of_steps)))
    Data_interaction_martix = np.zeros(((number_of_particles,number_of_particles)))
    Data_position = np.zeros((number_of_particles,2,number_of_steps))
    Data_angle_information = np.zeros((number_of_particles,number_of_particles,number_of_steps))
    Data_noises = np.zeros((number_of_particles,number_of_steps))


    ## Intiallize progressbar
    bar_widgets = ['Progress: ',progressbar.Percentage(), ' ', progressbar.Bar('#'),' ', progressbar.Timer(),
           ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
    pbar = progressbar.ProgressBar(widgets=bar_widgets, max_value=number_of_steps).start()

    ## Intialize file
    ### Folder name
    time_now = time.localtime()
    timeStr = time.strftime("%Y-%m-%d_%H-%M",time_now)
    folderName = timeStr + '_' + \
                str(number_of_particles) + 'particles_' + \
                str(number_of_influencer) + 'influencers_' + \
                str(wLF) + 'interaction_' + \
                str(number_of_steps) + 'steps_' + \
                str(format(eta/np.pi,'.1f')) + 'pi noises_' + \
                str(sense_radius) + 'radius_' + \
                str(size_of_arena) + 'size_' + \
                str(speed) + 'speed'#!!!!!!!!!!!!!!!!

    mypath = os.path.abspath(".")
    os.chdir(mypath)
    if not os.path.isdir(folderName):
        os.mkdir(folderName)
    os.chdir(folderName)
    ### File name 
    fileName = "vicsekData.hdf5"
    file = h5py.File(fileName, 'w-')


    #%%
    # update animation
    frames = []
    
    Data_interaction_martix=Wx.reshape((number_of_particles, number_of_particles))


    fig = plt.figure()  
    title_name = 'n=' + str(number_of_particles) + \
                ', $w_{lf}$=' +str(wLF) +  \
                ', steps=' + str(number_of_steps)  + \
                ', $\\eta$=' +str(format(eta/np.pi,'.1f')) +"$\\pi$"+  \
                ', r=' +str(sense_radius) +  \
                ', v=' + str(speed)
    
    ani = animation.FuncAnimation(fig=fig, func=simulation, frames=number_of_steps, interval=int(1/fps*1000), repeat=False)
    ani.save("./vicsek.mp4",fps=fps)
    # plt.show()

    pbar.finish()
    
    #%%
    # Save data
    file.create_dataset('angleSaved',           data=Data_theta,       compression='gzip', compression_opts=9)
    file.create_dataset('velocitySaved',        data=Data_velocity,    compression='gzip', compression_opts=9)
    file.create_dataset('positionSaved',        data=Data_position,    compression='gzip', compression_opts=9)
    file.create_dataset('adjacentSaved',        data=Data_adjacent_matrix,    compression='gzip', compression_opts=9)
    file.create_dataset('interactionSaved',     data=Data_interaction_martix, compression='gzip', compression_opts=9)
    file.create_dataset('angleInformationSaved',data=Data_angle_information, compression='gzip', compression_opts=9)
    file.create_dataset('NoisesSaved',          data=Data_noises, compression='gzip', compression_opts=9)

    numOfSteps = np.array([number_of_steps])
    timeResolution = np.array([time_resolution])
    wLF = np.array([wLF])
    numOfLeaders = np.array([number_of_influencer])
    noiseStrength = np.array([eta])
    numOfNodes = np.array([number_of_particles])
    sizeOfArena = np.array([size_of_arena])

    groupParameter = file.create_group("parameters")
    groupParameter.create_dataset('timeResolution',  data=timeResolution,     compression='gzip',   compression_opts=9)
    groupParameter.create_dataset('numOfSteps',         data=numOfSteps,            compression='gzip',   compression_opts=9)
    groupParameter.create_dataset('wLF',             data=wLF,                compression='gzip',   compression_opts=9)
    groupParameter.create_dataset('numOfLeaders',       data=numOfLeaders,          compression='gzip',   compression_opts=9)
    groupParameter.create_dataset('noiseStrength',   data=noiseStrength,      compression='gzip',   compression_opts=9)
    groupParameter.create_dataset('numOfNodes',        data=numOfNodes,           compression='gzip',   compression_opts=9)
    groupParameter.create_dataset('sizeOfArena',        data=sizeOfArena,           compression='gzip',   compression_opts=9)
    
    file.close()

    plt.close()
    os.chdir("..") 