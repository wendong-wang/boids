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


def tanh(x:np.ndarray):

    # tanh 
    # nonlinear_x = (2/(1+np.exp(-2*x))-1)*np.pi 
    # nonlinear_x=np.tanh(x)*np.pi #
    # nonlinear_x=np.tanh(x*2/np.pi)*np.pi #
    
    # singmoid 
    # nonlinear_x = (1/(1+np.exp(-x))-1/2)*2*np.pi
    
    # linear
    nonlinear_x = x
    return nonlinear_x


class Vicsek():
    '''
    class of Vicsek model, each class is a group of Vicsek model

    Parameters
    --
    sizeOfArena: float ( default: )
        the linear size of the squrae shape cell where simulations are carried out. 
    number: int  ( default: )
        the number of units. 
    numOfLeaders: int  ( default: )
        the number of leader. 
    wLF: float  ( default: )
        the interaction that leaders exert on followers. 
    speed: float ( default: )
        speed of the units(constant temporarily).
    senseRadiu: float ( default: )
        the radius that an unit can feel.
    noises: float ( default: )
        the strength of noises
    name: str (default: vicsek)
        name of viscek model.

    >>> hello vicsek

    '''
    '''
    Notation
    --
        # l: the simulations are carried out in a square shape cell of linear size L 1x1 场地大小
        # n: n units 1x1 个数
        # v: value of the velocities of the units  1x1 速度大小
        # Theta: directions nx1 角度(每个unit运动的不同)
        # Ve: velocities of the units nx2
        # Po: matrix of the positions of the units n x 2 x numOfSteps 位置矩阵
        # r: the radius that an unit can feel 1x1 所感知的半径
        # eta: the strength of the noises 1x1 噪声
        # Wx: the matrix of interaction strength nxn 
    大小写规范问题:小写为数值，大写开头为矩阵

    '''
    def __init__(self, sizeOfArena: float, number: int, numOfLeaders: int, wLF: list[float], speed: float, senseRadius: float,noises: float, name: str = "vicsek" ) -> None:
        self.name = name
        print(name,"is born")
        #####
        self.parameters_Init(sizeOfArena=sizeOfArena,number = number,numOfLeaders = numOfLeaders,wLF = wLF,speed = speed, senseRadius = senseRadius, noises = noises) #init parameters
        
        pass
    


    def parameters_Init(self, sizeOfArena: float, number:int, numOfLeaders: int, wLF: list[float], speed: float, senseRadius: float, noises: float) -> None:
        '''
        Init parameters

        Parameters
        --
        sizeOfArena: float ( default: )
           the linear size of the squrae shape cell where simulations are carried out. 
        number: int  ( default: )
            the number of units. 
        numOfLeaders: int  ( default: )
            the number of leader. 
        wLF: float  ( default: )
            the interaction that leaders exert on followers. 
        speed: float ( default: )
            speed of the units(constant temporarily).
        senseRadiu: float ( default: )
            the radius that an unit can feel.
        noises: float ( default: )
            the strength of noises.

        '''
        self.l = sizeOfArena
        self.n = number
        self.v = speed
        self.r = senseRadius
        self.eta = noises
        # init Position P
        rng = np.random.default_rng()
        self.Po = rng.random((self.n,2))*self.l # random()-> float interval [0,1]
        # init angle   Theta        0~2*pi but what is 
        self.Theta = rng.random((self.n,1))*2*np.pi 
        # init Velocity  V    
        self.Ve = np.hstack((self.v*np.cos(self.Theta),self.v*np.sin(self.Theta)))
        # init interaction strength
        self.Wx = np.ones((self.n,self.n))
        self.indexOfLeader = numOfLeaders -1 # self.indexOfLeader is the max labels of leaders. 
                                      # If there is one leader,  self.indexOfLeader = 1-1=0, meaning that [0] is a leader.
                                      # If there are three, self.indexOfLeader = 3-1 = 2, meaning that [0][1][2] are leaders. 
          
        self.wLF =wLF
        # wLF = [self.wLF, 1] # for two leader cases
        # self.Wx[self.indexOfLeader+1:,0:self.indexOfLeader+1] = wLF  # leader's effect on followers

        self.Wx[0:self.indexOfLeader+1,self.indexOfLeader+1:] = 0 # followers' effect on leader
        self.Wx[0:self.indexOfLeader+1,0:self.indexOfLeader+1] = 0 #  leader on leader
        self.Wx[self.indexOfLeader+1:,self.indexOfLeader+1:] =0 # ; follower on follower
        followerIndex = np.arange(self.indexOfLeader+1,self.n)
        for i in followerIndex: #follower on itself
            self.Wx[i,i]=1

        for i in range(self.indexOfLeader+1): #leader on itself
            self.Wx[self.indexOfLeader+1:,i] = wLF[i]  # leader's effect on followers # leader 之间没有相互影响
            self.Wx[i,i]=wLF[i]
                # print(i)
        print("wx\n", self.Wx[:5,:5])
        
        # force the first leader's initial position and velocity
        # temporarily for rotation
        self.leader_Init(indexOfLeader = self.indexOfLeader)

        self.frameFlag = True
        self.rstar = [3, -1]

        pass



#================== simulation part

    def start(self,numOfSteps) -> None:
        '''
        start the simulation
        
        Parameters:
        numOfSteps: int (default:)
            the number of steps.
        '''
        self.simulation_Init(numOfSteps) # init file and folder 
        self.InteractionSaved[:,:] = self.Wx.reshape((self.n, self.n ))
        fig = plt.figure() #


        ani = animation.FuncAnimation(fig=fig, func=self._move, frames=self.numOfSteps, interval=int(1/self.fps*1000), blit=False, repeat=False) #
        ani.save('./vicsek.mp4',fps=self.fps)
        
        self.pbar.finish()
        print(":) \"Video is saved successfully.\"")
        self.data_Save()
        print(":) \"Data is saved successfully.\"")
        # plt.show()
        plt.close()# 否则一次画太多的图了

    def _move(self, frameNumber):
        '''
        update the animation, plot

        Parameters:
        ---
            frameNumber: 
                the number of the frame, 
        '''
        self.now = frameNumber
        if self.frameFlag == True:
            self.frameFlag = False
            return

        
        
        indexOfLeader = self.indexOfLeader
        Ve = self.Ve[:,:]
        Po = self.Po[:,:]
        Theta = self.Theta[:]
        colorBoard = ['b','c','m','y','r','g','k','olive','purple','turquoise','lightpink']
        n = self.n
        # PoA = np.hstack(((Po[:,0]*Ax[:,0]).reshape(n,1), (Po[:,1]*Ax[:,0]).reshape(n,1)))
        # print(self.now)
        # save data

        self.ThetaSaved[:,self.now] = Theta.reshape((self.n, ))
        self.VelocitySaved[:,:,self.now] = Ve.reshape((self.n, 2, ))
        self.PositionSaved[:,:,self.now] = Po.reshape((self.n, 2, ))
        



        self.pbar.update(self.now+1)
        plt.cla()

        titleName = 'n=' + str(self.n) + \
                    ', $w_{lf}$=' +str(self.wLF) +  \
                    ', step=' + str(self.numOfSteps)  + \
                    ', $\\eta$=' +str(format(self.eta/np.pi,'.1f')) +"$\\pi$"+  \
                    ', L=' +str(self.l) +  \
                    ', r=' +str(self.r) +  \
                    ', v=' + str(self.v) + \
                    ', $r^{*}$=' + str(self.rstar[0:self.indexOfLeader+1]) 
        
        plt.title(titleName)
        plt.xlim((0, self.l))
        plt.ylim((0, self.l))
        now = str(self.now)
        plt.xlabel(xlabel="$t_n = $"+now,size = 15) # 真实与显示可能略微有些误差
        # 画与领导节点相邻的线 draw the lines between leaders and the followers affected by the leaders
        # try:
        #     for j in range(indexOfLeader+1):
        #         PoA = np.hstack(((Po[:,0]*self.Ax[:,j]).reshape(n,1), (Po[:,1]*self.Ax[:,j]).reshape(n,1)))
        #         for i in PoA:
        #             if i[0] !=0. and i[1] != 0.:
        #                 # plt.plot(np.array([PoA[0,0],i[0]]),np.array([PoA[0,1],i[1]]),color='blue')
        #                 plt.plot(np.array([PoA[j,0],i[0]]),np.array([PoA[j,1],i[1]]),color=colorBoard[j],linestyle='--',alpha = 0.5,linewidth=0.8)
        # except:
        #     pass
        
        
        # draw the quivers
        # plt.quiver(Po[0:indexOfLeader+1,0],Po[0:indexOfLeader+1,1],Ve[0:indexOfLeader+1,0],Ve[0:indexOfLeader+1,1],color = 'r',width = 0.005,scale = 50,scale_units='width') # init position
        if indexOfLeader>=0:
            plt.quiver(Po[0:indexOfLeader+1,0],Po[0:indexOfLeader+1,1],np.cos(Theta[0:indexOfLeader+1]),np.sin(Theta[0:indexOfLeader+1]),color = 'r',width = 0.005) # init position
        # if self.n >self.indexOfLeader+1:
        #     plt.quiver(Po[indexOfLeader+1,0],Po[indexOfLeader+1,1],Ve[indexOfLeader+1,0],Ve[indexOfLeader+1,1],color = 'b',width = 0.005)
        #     if self.n >self.indexOfLeader+2:
        #         plt.quiver(Po[indexOfLeader+2:,0],Po[indexOfLeader+2:,1],Ve[indexOfLeader+2:,0],Ve[indexOfLeader+2:,1],width = 0.005)
        
        if self.n >self.indexOfLeader+1:
        #     plt.quiver(Po[indexOfLeader+2:,0],Po[indexOfLeader+2:,1],Ve[indexOfLeader+2:,0],Ve[indexOfLeader+2:,1],width = 0.005)
            plt.quiver(Po[indexOfLeader+1:,0], Po[indexOfLeader+1:,1], np.cos(Theta[indexOfLeader+1:]), np.sin(Theta[indexOfLeader+1:]), width = 0.005)
        # plt.quiver(Po[:,0],Po[:,1], self.Ve[:,0], self.Ve[:,1])

        # 编号 serial number
        for i in range(np.shape(Po)[0]):
            plt.text(Po[i,0]-0.1, Po[i,1]+0.1, str(i), color="brown")
        
        pass
        
        #绘制轨迹 draw the previous trajectory
        trajL = 15 # trajectory length
        scatterSize = np.ones((self.n,trajL))*0.5 # trajectory size
        if self.now > trajL:
            plt.scatter(self.PositionSaved[:,0,self.now-trajL:self.now], self.PositionSaved[:,1,self.now-trajL:self.now], marker='.', s=scatterSize[:,:])#画轨迹点
        else:
            plt.scatter(self.PositionSaved[:,0,:self.now], self.PositionSaved[:,1,:self.now], marker='.', s=scatterSize[:,:self.now])#画轨迹点


        Po, Theta, Ax, Noises= self.update()
        self.Ax = Ax
        self.AdjacentSaved[:,:,self.now] = Ax.reshape((self.n, self.n, ))

        # if self.now == 0:
        #     self.NoisesSaved[:,self.now] = np.zeros(self.n)#初始没有噪声
        # if self.now != (self.numOfSteps-1):
        #     self.NoisesSaved[:,self.now+1] = Noises.reshape((self.n, ))
        self.NoisesSaved[:,self.now] = np.zeros(self.n)





    def simulation_Init(self,numOfSteps):
        '''
        Init things which are used for simulation

        Parameters:
        --
        numOfSteps: int (default:)
            the number of steps.

        Section
        --
        >>> step (time) init
        >>> space init
        >>> progressbar init
        >>> file init: folder and hdf5 file 

        '''

        # step (time) init
        #------------
        # self.simulationTime = 10 # the duration of the simulation, unit: second
        # self.step = 0.1 # the duration of the step
        # self.numOfSteps = int(self.simulationTime/self.step)

        self.numOfSteps = numOfSteps
        self.now = 0 # record what the step is now, start from 0~(numOfSteps-1)
        '''
        self.timeResolution:
            The time period between adjacent video frames is 1/20.
        Our vedio is 20fps meaning that 20 frames per second. Therefore,
        the time period between adjacent video frames is 1/20
        '''
        self.fps = 20 
        self.timeResolution = 1/self.fps  # 

        # space init
        #----------------
        # space for data to be saved
        self.ThetaSaved = np.zeros((self.n, self.numOfSteps))
        self.VelocitySaved = np.zeros((self.n,2,self.numOfSteps))
        self.AdjacentSaved = np.zeros(((self.n,self.n,self.numOfSteps)))
        self.InteractionSaved = np.zeros(((self.n,self.n)))
        self.PositionSaved = np.zeros((self.n,2,self.numOfSteps))
        self.angleInformationSaved = np.zeros((self.n,self.n,self.numOfSteps))
        self.NoisesSaved = np.zeros((self.n,self.numOfSteps))
        # progressbar init
        #------------
        widgets = ['Progress: ',progressbar.Percentage(), ' ', progressbar.Bar('#'),' ', progressbar.Timer(),
           ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, max_value=self.numOfSteps).start()
        # self.pbar = progressbar.ProgressBar().start()

        # file init
        #-----------
        # init folder
        time_now = time.localtime()
        timeStr = time.strftime("%Y-%m-%d_%H-%M",time_now)
        self.folderName = timeStr + '_' + \
                    str(self.n) + 'particles_' + \
                    str(self.indexOfLeader+1) + 'leaders_' + \
                    str(self.wLF) + 'interaction_' + \
                    str(self.numOfSteps) + 'steps_' + \
                    str(format(self.eta/np.pi,'.1f')) + 'pi noises_' + \
                    str(self.r) + 'radius_' + \
                    str(self.l) + 'size_' + \
                    str(self.rstar[0:self.indexOfLeader+1]) + 'rstar_' + \
                    str(self.v) + 'speed'#!!!!!!!!!!!!!!!!

        mypath = os.path.abspath(".")
        os.chdir(mypath)
        if not os.path.isdir(self.folderName):
            os.mkdir(self.folderName)
        os.chdir(self.folderName) 


        # init file
        fileName = "vicsekData.hdf5"
        self.file = h5py.File(fileName, 'w-')

    def data_Save(self):
        '''
        save data
        '''

        self.file.create_dataset('angleSaved',       data=self.ThetaSaved,       compression='gzip', compression_opts=9)
        self.file.create_dataset('velocitySaved',    data=self.VelocitySaved,    compression='gzip', compression_opts=9)
        self.file.create_dataset('positionSaved',    data=self.PositionSaved,    compression='gzip', compression_opts=9)
        self.file.create_dataset('adjacentSaved',    data=self.AdjacentSaved,    compression='gzip', compression_opts=9)
        self.file.create_dataset('interactionSaved', data=self.InteractionSaved, compression='gzip', compression_opts=9)
        self.file.create_dataset('angleInformationSaved', data=self.angleInformationSaved, compression='gzip', compression_opts=9)
        self.file.create_dataset('NoisesSaved', data=self.NoisesSaved, compression='gzip', compression_opts=9)
   
        numOfSteps = np.array([self.numOfSteps])
        timeResolution = np.array([self.timeResolution])
        wLF = np.array([self.wLF])
        numOfLeaders = np.array([self.indexOfLeader+1])
        noiseStrength = np.array([self.eta])
        numOfNodes = np.array([self.n])
        sizeOfArena = np.array([self.l])

        groupParameter = self.file.create_group("parameters")
        groupParameter.create_dataset('timeResolution',  data=timeResolution,     compression='gzip',   compression_opts=9)
        groupParameter.create_dataset('numOfSteps',         data=numOfSteps,            compression='gzip',   compression_opts=9)
        groupParameter.create_dataset('wLF',             data=wLF,                compression='gzip',   compression_opts=9)
        groupParameter.create_dataset('numOfLeaders',       data=numOfLeaders,          compression='gzip',   compression_opts=9)
        groupParameter.create_dataset('noiseStrength',   data=noiseStrength,      compression='gzip',   compression_opts=9)
        groupParameter.create_dataset('numOfNodes',        data=numOfNodes,           compression='gzip',   compression_opts=9)
        groupParameter.create_dataset('sizeOfArena',        data=sizeOfArena,           compression='gzip',   compression_opts=9)
        
        
        self.file.close()
        os.chdir("..")
        pass



#=================== update data
    def alignment(self, WxA, distance):
        '''
        Compute the alignment. Weighted average the velocity of adjacent units.
        
        Parameters:
        ---
        WxA: interaction matrix .* adjacent node matrix

        Return:
        ---
        The result direction of alignment, whose size is (n x 2) . Each row of the matrix, (x,y), is unitized. 
        '''
        Ve = self.Ve
        n = self.n
        senseRadius = self.r


        Theta = self.Theta[:]

        WAlignment   = np.ones((n,n)) # aligment weight matrix

        detaTheta = -Theta.reshape((n,1))+Theta.reshape((1,n)) # i, j is \theta j - theta i
        #dtatTheta \in [-2pi,2pi]
        
        detaTheta = detaTheta*((detaTheta<=np.pi)*(detaTheta>-np.pi)) + (((detaTheta-2*np.pi)*(detaTheta>np.pi))) + (((detaTheta+2*np.pi)*(detaTheta<=-np.pi)))#也可以用取余来操作
        #dtatTheta \in [-pi,pi]

        ThetaA = WxA*WAlignment*detaTheta #\inc\theta_i
        total_weight = np.sum(WxA,axis=1).reshape((n,1))
        ThetaA = np.divide(ThetaA, total_weight, where = total_weight>0)
        # print("hollo1\n",ThetaA)
        nonlinearThetaA = tanh(ThetaA)
        # print("hollo2\n",nonlinearThetaA)
        ThetaASum = np.sum(nonlinearThetaA,axis=1)
        ThetaInfluence = nonlinearThetaA
        
        self.angleInformationSaved[:,:,self.now] = ThetaInfluence[:,:]
        return ThetaASum

  

    def update(self):
        '''
        # Main logic
        
        update the position.        
        
        Parameters:

        '''
        
        dx = np.subtract.outer(self.Po[:, 0], self.Po[:, 0])
        dy = np.subtract.outer(self.Po[:, 1], self.Po[:, 1]) 
        # distance = np.hypot(dx, dy)

        # periodic boundary                                          # this column increase the robust of the condition
        # distanceTemp = np.zeros((self.n,self.n))                    
        # distanceTemp += (dy > self.l/2) * (np.abs(dx)< self.l/2) * (np.hypot(0-dx,self.l-dy)<self.r)       *np.hypot(0-dx,self.l-dy)
        # distanceTemp += (dy > self.l/2) * (dx< -self.l/2)        * (np.hypot(-self.l-dx,self.l-dy)<self.r) *np.hypot(-self.l-dx,self.l-dy)
        # distanceTemp += (dy > self.l/2) * (dx> self.l/2)         * (np.hypot(self.l-dx,self.l-dy)<self.r)  *np.hypot(self.l-dx,self.l-dy)
        # # distanceTemp += (dy < self.l/2) * (dx> self.l/2)         * (np.hypot(self.l-dx,0-dy)<self.r)       *np.hypot(self.l-dx,0-dy)        # for r < l/2 there is no problem， but for r>l/2 there will be a risk
        # distanceTemp += (abs(dy) < self.l/2) * (dx> self.l/2)    * (np.hypot(self.l-dx,0-dy)<self.r)       *np.hypot(self.l-dx,0-dy)

        # distanceTemp += distanceTemp.T
        # distance = (distanceTemp==0)*distance + distanceTemp

        dx = (abs(dx)>(self.l/2))*(self.l-abs(dx))+(abs(dx)<=(self.l/2))*abs(dx)
        dy = (abs(dy)>(self.l/2))*(self.l-abs(dy))+(abs(dy)<=(self.l/2))*abs(dy)
        distance = np.hypot(dx, dy)

        
        Ax = (distance >= 0) * (distance <= self.r) # >=0是包括自己 

        # print(Ax)
        di = np.maximum(Ax.sum(axis=1), 1) #.reshape(self.n,1)
        Dx = np.diag(di)
        # print(Dx)
        Lx = Dx-Ax
        Id = np.identity(self.n)
       
       
        WxA = Ax * self.Wx
        # noise
        rng = np.random.default_rng()

        Noises = rng.random((self.n,1))*self.eta - self.eta/2 #[-eta/2, eta/2]

        # calculate
        
        
        #angle

        # # different interaction modes
        # #A    
        # dTheta = self.alignment(WxA=WxA, distance=distance)#alignment                            #alignment
        # self.Theta = self.Theta + dTheta.reshape((self.n,1))

        #A'
        if self.now%2==0:
            rng = np.random.default_rng()
            self.Theta = rng.random((self.n,1))*2*np.pi - np.pi/2
            dTheta = self.alignment(WxA=WxA, distance=distance) #alignment
            self.Theta = self.Theta + dTheta.reshape((self.n,1))
        else:
            dTheta = self.alignment(WxA=WxA, distance=distance) #alignment
            self.Theta = self.Theta + dTheta.reshape((self.n,1))

        
        # add noises
        self.Theta = self.Theta + Noises
        # self.Theta[:self.indexOfLeader+1] += Noises[:self.indexOfLeader+1]
        # self.Theta[:self.indexOfLeader+1] -= Noises[:self.indexOfLeader+1] # exclude the noise of leader

        # leader control
        self.Theta= self.update_Leader_Control()
        # interval range of angle->[0,2*pi)
        self.Theta = np.mod(self.Theta, 2 * np.pi)
        
        # speed remains unchanged
        # velocity
        self.Ve = np.hstack((self.v*np.cos(self.Theta),self.v*np.sin(self.Theta)))
        # self.Ve = np.hstack((self.v*np.cos(self.Theta)+Noisesx,self.v*np.sin(self.Theta)+Noisesy))########################


        self.Theta = np.arctan2(self.Ve[:,1].reshape(self.n,1),self.Ve[:,0].reshape(self.n,1))###############3
        # interval range of angle->[0,2*pi)
        self.Theta = np.mod(self.Theta, 2 * np.pi)########################

        # position
        # self.realX = self.realX +self.Ve[0:self.indexOfLeader+1,0] * self.timeResolution#!!!!!!!跟sin有关
        self.Po  = self.Po + self.Ve * self.timeResolution # 1 means time
        self.Po = np.mod(self.Po, self.l)
        
        return self.Po, self.Theta, Ax, Noises



#=====================================================leader init
    def leader_Init(self,indexOfLeader):
        '''
        leader init
        init the position, the velocities, and the angle.

        Parameters:
        ---
        indexOfLeader: the largest index of the leader, start from 0.

        '''

        indexOfLeader = self.indexOfLeader
        fps = 20
        timeResolution = 1/fps
        v = self.v
        indexOfLeader=self.indexOfLeader
        

        #two rotation 
        # rStar = [1.5,-1.5]
        # self.Po[0,:]=[self.l/4+np.abs(rStar[0]),self.l/2]
        # self.Po[1,:]=[self.l*3/4-np.abs(rStar[1]),self.l/2]

        # self.Ve[0,:]=[0,1]
        # self.Ve[1,:]=[0,1]
        #     #follower design
        # self.Po[2,:]= [self.l*2/5,self.l/2]
        
        # self.Po[3,:]= [self.l/2,0]
        # self.Ve[3,:]= [0,1]





        #     self.Theta[i] = np.arctan2(self.Ve[i,1].reshape(1,1),self.Ve[i,0].reshape(1,1))
        
        # self.realX = self.Po[0:indexOfLeader+1,0]
        # Rotation
        # r = 3
        # dTheta = 2*np.pi/(indexOfLeader+1)
        # for i in range(indexOfLeader+1):
        #     self.Po[i,:]=[self.l/2+r*np.cos(np.pi/2+dTheta*i),self.l/2+r*np.sin(np.pi/2+dTheta*i)]
        #     self.Ve[i,:]=[-3*np.sin(np.pi/2+dTheta*i),3*np.cos(np.pi/2+dTheta*i)]
        #     self.Theta[i] = np.arctan2(self.Ve[i,1].reshape(1,1),self.Ve[i,0].reshape(1,1))
        
        
        # self.idealPosition =  self.Po[0:indexOfLeader+1,:]
        # self.idealTheta = self.Theta[0:indexOfLeader+1]
        # self.idealVe = self.Ve[0:indexOfLeader+1,:]

        #--------------
        # line
        # #nothing
        #--------------
        # sin
        # fakeAmplitude = 2
        # frequency = 0.5
        # # Theta[0:self.indexOfLeader+1] = np.arctan(amplitude *np.cos(2*np.pi*frequency*t))
        # self.Theta[0:self.indexOfLeader+1] = np.arctan(fakeAmplitude *np.cos(2*np.pi*frequency*0))


        # a = 0
        # self.Theta[0:self.indexOfLeader+1] = ((-1)**(int(a/np.pi)) *a)%np.pi -np.pi/2

            #new sin
        # x = self.realX
        # tau = timeResolution
        # self.Theta[0:indexOfLeader+1] = np.arctan(np.cos(x+((v*tau**2)/(1+np.cos(x)**2))**0.5))
        pass

    def update_Leader_Control(self):        
        '''
        Control the movement of leader.

        Parameters
        --
        Noises: Noises

        '''
        # at the beginning the self.now == 0
        # we are under the assuption that all the leaders are doing the same movement locally.
        # we describe the movement with parametric equation about time t.

        Theta = self.Theta
        indexOfLeader = self.indexOfLeader 
        timeResolution = self.timeResolution
        v = self.v
        Po = self.Po
        l = self.l
      

        # if self.now>2000:
        #     Theta[0:indexOfLeader+1] = Theta[0:indexOfLeader+1] -Noises[0:indexOfLeader+1] # eliminate the noises 

        #     # rotation
            
        #     # r = self.r/np.sin(np.pi/self.n)
        #     # rStar = [-1,1]
        #     # for i in range(indexOfLeader+1):
        #     #     Theta[i] = Theta[i] + self.v*timeResolution/rStar[i] + Noises[i]# dtheta = v*dt/r
        #     r=3
        #     Theta[0:indexOfLeader+1] = Theta[0:self.indexOfLeader+1] + self.v*timeResolution/r + Noises[0:indexOfLeader+1]# dtheta = v*dt/r
        

        # milling
            


        # rStar = self.rstar[:]
        # for i in range(indexOfLeader+1):
        #     Theta[i] = Theta[i] + self.v*timeResolution/rStar[i]# dtheta = v*dt/、

        
        
        # idealTheta = self.idealTheta
        # idealPosition = self.idealPosition
        # idealVe = self.idealVe
        # idealTheta = idealTheta + self.v*timeResolution/r
        # idealVe = np.hstack((self.v*np.cos(idealTheta),self.v*np.sin(idealTheta)))
        # idealPosition = idealPosition + idealVe* self.timeResolution 
        # alphaTheta = np.arctan2((idealPosition[:,1]- Po[0:self.indexOfLeader+1,1]),(idealPosition[:,0]- Po[0:self.indexOfLeader+1,0]))

        # Theta[0:self.indexOfLeader+1] = Theta[0:self.indexOfLeader+1]+ np.sin(alphaTheta.reshape(self.indexOfLeader+1,1)-Theta[0:self.indexOfLeader+1])+ Noises[0:indexOfLeader+1]
        # # Theta[0:self.numOfLeaders+1] = Theta[0:self.numOfLeaders+1]+ alphaTheta-Theta[0:self.numOfLeaders+1]+ Noises[0:numOfLeaders+1]

        # self.idealTheta = idealTheta
        # self.idealPosition =idealPosition
        # self.idealVe = idealVe



        #----------------
        #line



        # Theta[0:self.indexOfLeader+1] = np.pi/4
        
        #------------
        # sin wave
        
        # t = self.now *self.timeResolution
        # fakeAmplitude = 2
        # frequency = 0.5
        # # Theta[0:self.indexOfLeader+1] = np.arctan(amplitude *np.cos(2*np.pi*frequency*t))

        # Theta[0:self.indexOfLeader+1] = np.arctan(fakeAmplitude *np.cos(2*np.pi*frequency*t))
        
            #new sin
        # x = self.realX
        # tau = timeResolution
        # Theta[0:self.indexOfLeader+1] = np.arctan(np.cos(x+((v*tau**2)/(1+np.cos(x)**2))**0.5))


        return Theta
    


if __name__ == "__main__":

    '''
    please check what type the module is
    now: different leader, different w_LF 
    '''
    numberOfLeaders = 1 #number Of Leaders 3
    numberofUnits = 49 + numberOfLeaders #number Of Units, 10 is the number of followers 7
    sizeArea = 10
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=5, speed=2, senseRadius=5 , noises=0.3*np.pi, name="vicsek").start(numOfSteps=1000)# noises were added to the velocity
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=5, speed=2, senseRadius=3 , noises=2/3*np.pi, name="vicsek").start(numOfSteps=131072)# noises were added to the velocity
    noises = 0.1*np.pi
    # print(noises/3.14*180)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.1, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_0.1").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.3, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_0.3").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.5, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_0.5").start(numOfSteps=131072)  
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.7, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_0.7").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.9, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_0.9").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.1, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_1.1").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.3, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_1.3").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.5, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_1.5").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=2.5, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_2.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=3.5, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_3.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=4.5, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_4.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=6.0, speed=2, senseRadius=4.5, noises=noises,  name="vicsek_6.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=10., speed=2, senseRadius=4.5, noises=noises,  name="vicsek_10.").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.2, speed=2, senseRadius=4.5, noises=noises, name="vicsek_0.2").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.4, speed=2, senseRadius=4.5, noises=noises, name="vicsek_0.4").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.6, speed=2, senseRadius=4.5, noises=noises, name="vicsek_0.6").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=0.8, speed=2, senseRadius=4.5, noises=noises, name="vicsek_0.8").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.0, speed=2, senseRadius=4.5, noises=noises, name="vicsek_1.0").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.2, speed=2, senseRadius=4.5, noises=noises, name="vicsek_1.2").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.4, speed=2, senseRadius=4.5, noises=noises, name="vicsek_1.4").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=2.0, speed=2, senseRadius=4.5, noises=noises, name="vicsek_2.0").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=3.0, speed=2, senseRadius=4.5, noises=noises, name="vicsek_3.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=4.0, speed=2, senseRadius=4.5, noises=noises, name="vicsek_4.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=5.0, speed=2, senseRadius=4.5, noises=noises, name="vicsek_5.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=8.0, speed=2, senseRadius=4.5, noises=noises, name="vicsek_8.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=15., speed=2, senseRadius=4.5, noises=noises, name="vicsek_15.").start(numOfSteps=131072) 
    
             
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.0, speed=2, senseRadius=5, noises=noises, name="vicsek_1.0").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.2, speed=2, senseRadius=5, noises=noises, name="vicsek_1.2").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.4, speed=2, senseRadius=5, noises=noises, name="vicsek_1.4").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=2.0, speed=2, senseRadius=5, noises=noises, name="vicsek_2.0").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=3.0, speed=2, senseRadius=5, noises=noises, name="vicsek_3.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=4.0, speed=2, senseRadius=5, noises=noises, name="vicsek_4.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=5.0, speed=2, senseRadius=5, noises=noises, name="vicsek_5.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=8.0, speed=2, senseRadius=5, noises=noises, name="vicsek_8.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=15., speed=2, senseRadius=5, noises=noises, name="vicsek_15.").start(numOfSteps=131072) 
    
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.1, speed=2, senseRadius=5, noises=noises,  name="vicsek_1.1").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.3, speed=2, senseRadius=5, noises=noises,  name="vicsek_1.3").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=1.5, speed=2, senseRadius=5, noises=noises,  name="vicsek_1.5").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=2.5, speed=2, senseRadius=5, noises=noises,  name="vicsek_2.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=3.5, speed=2, senseRadius=5, noises=noises,  name="vicsek_3.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=4.5, speed=2, senseRadius=5, noises=noises,  name="vicsek_4.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=6.0, speed=2, senseRadius=5, noises=noises,  name="vicsek_6.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=10., speed=2, senseRadius=5, noises=noises,  name="vicsek_10.").start(numOfSteps=131072) 
   
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=20., speed=2, senseRadius=5, noises=(0.3)*np.pi, name="vicsek_20.").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=25., speed=2, senseRadius=5, noises=(0.3)*np.pi, name="vicsek_25.").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=30., speed=2, senseRadius=5, noises=(0.3)*np.pi, name="vicsek_30.").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=40., speed=2, senseRadius=5, noises=(0.3)*np.pi, name="vicsek_40.").start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=3, noises=0.6*np.pi, name="vicsek_3.0").start(numOfSteps=1000) 

    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=4, noises=0.4*np.pi, name="vicsek_3.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=3, noises=1.0*np.pi, name="vicsek_3.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=3, noises=1.4*np.pi, name="vicsek_3.0").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=3, noises=2.0*np.pi, name="vicsek_3.0").start(numOfSteps=131072) 
   
   #to borden the synergistic 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0,5.0], speed=2, senseRadius=8, noises=0.3*np.pi, name="vicsek").start(numOfSteps=1000) 

    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0,5.0], speed=2, senseRadius=8, noises=0.3*np.pi, name="vicsek").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0,5.0], speed=2, senseRadius=8, noises=0.3*np.pi, name="vicsek").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0,5.0], speed=2, senseRadius=8, noises=0.3*np.pi, name="vicsek").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=0.3*np.pi, name="vicsek").start(numOfSteps=1000) 


   # varying with eta
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*0.1, name="vicsek_.1").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*0.5, name="vicsek_.5").start(numOfSteps=131072) 
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*0.8, name="vicsek_.8").start(numOfSteps=131072) 
# 
    for i in range(7):
        # Vicsek(sizeOfArena= sizeArea, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(i)*0.1, name="vicsek_ata_"+str((i)*0.1)).start(numOfSteps=131072)
        # Vicsek(sizeOfArena= sizeArea, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(i+7)*0.1, name="vicsek_ata_"+str((i+7)*0.1)).start(numOfSteps=131072)
        Vicsek(sizeOfArena= sizeArea, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(i+14)*0.1, name="vicsek_ata_"+str((i+14)*0.1)).start(numOfSteps=131072)
        # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=5, speed=2, senseRadius=5, noises=np.pi*(i+16)*0.1, name="vicsek_ata_"+str((i+16)*0.1)).start(numOfSteps=131072)
    
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(4)*0.1, name="vicsek_ata_"+str((3)*0.1)).start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(7)*0.1, name="vicsek_ata_"+str((4)*0.1)).start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(9)*0.1, name="vicsek_ata_"+str((5)*0.1)).start(numOfSteps=131072)
    # # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(11)*0.1, name="vicsek_ata_"+str((6)*0.1)).start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(13)*0.1, name="vicsek_ata_"+str((13)*0.1)).start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(16)*0.1, name="vicsek_ata_"+str((16)*0.1)).start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(17)*0.1, name="vicsek_ata_"+str((17)*0.1)).start(numOfSteps=131072)
    # Vicsek(sizeOfArena= 10.0, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[5.0], speed=2, senseRadius=5, noises=np.pi*(19)*0.1, name="vicsek_ata_"+str((19)*0.1)).start(numOfSteps=131072)
   
    
    
    # numberofUnits = 200 + numberOfLeaders #number Of Units, 10 is the number of followers 7
    # sizeArea = 14.1

    # for i in range(7):
    #     # Vicsek(sizeOfArena= sizeArea, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[1], speed=2, senseRadius=1, noises=np.pi*(i)*0.1, name="vicsek_ata_"+str((i)*0.1)).start(numOfSteps=8000)
    #     Vicsek(sizeOfArena= sizeArea, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[1], speed=2, senseRadius=1, noises=np.pi*(i+7)*0.1, name="vicsek_ata_"+str((i+7)*0.1)).start(numOfSteps=8000)
    #     Vicsek(sizeOfArena= sizeArea, number=numberofUnits, numOfLeaders=numberOfLeaders, wLF=[1], speed=2, senseRadius=1, noises=np.pi*(i+14)*0.1, name="vicsek_ata_"+str((i+7)*0.1)).start(numOfSteps=8000)

    
'''
[1] Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I. & Shochet, O. Novel Type of Phase Transition in a System of Self-Driven Particles. Phys. Rev. Lett. 75, 1226-1229 (1995).
[2] Cucker, F. & Smale, S. Emergent Behavior in Flocks. IEEE Trans. Automat. Contr. 52, 852-862 (2007).
[3] Sattari, S. et al. Modes of information flow in collective cohesion. SCIENCE ADVANCES 14 (2022).

'''