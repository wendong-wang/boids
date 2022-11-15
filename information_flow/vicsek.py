# vicsek model simulation [1]
# Under developing
# authority: Pang Jiahuan
# start time: 2022/11/7
# last time: 2022/11/15
# end time: ~
# python: 3.6

'''
vicsek model version description:
    改成[3]中的模型，描述参考[2]，加入数据处理部分

相对与上一版改进的说明：
    取消了惯性，改用速度取平均,加入数据保存,但是这里保存只是考虑了角度值
    结构也大改，将仿真作为类的一个方法，并将仿真数据与模型参数数据进行隔离
    数据：角度，步数
'''
#考虑：
#   1. 数学计算采用[2]的方式？还是逐个元素进行相应的计算
#   2. 加速度的表示，注意量纲dimension (暂时不考虑惯性了)
#   3. 各种区间边界的开闭确定： 角度， 噪声
#   4. HDF5数据的结构 (暂不建立group, only dataset)
#   5. 能否把控时间，而不只是仅仅考虑多少步，simulationTime step（暂时删除）存在的意义
#   6. 把视频保存放进data_Save()函数
#   7. 全大写的变量定义为常量，这里希望进行略微的修改(为了步保证变量，后面加入小写)
#   ·。版本修改3.8->3.6 (numpy 在python3.8 下的一些函数竟然在 python3.6中也好使？虽然代码上没有了相应的提示符 weird)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time
import os
import h5py
import progressbar

class Vicsek():
    '''
    class of Vicsek model, each class is a group of Vicsek model

    Parameters
    --
    >>> sizeOfArena: float ( default: )
            the linear size of the squrae shape cell where simulations are carried out. 
    >>> number: int  ( default: )
            the number of units. 
    >>> speed: float ( default: )
            speed of the units(constant temporarily).
    >>> senseRadiu: float ( default: )
            the radius that an unit can feel.
    >>> noises: float ( default: )
            the strength of noises
    >>> name: str (default: vicsek)
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
        # Vo: velocities of the units nx2
        # Po: matrix of the positions of the units n x 2 x stepNum 位置矩阵
        # r: the radius that an unit can feel 1x1 所感知的半径
        # yeta: the strength of the noises 1x1 噪声
    大小写规范问题:小写为数值，大写开头为矩阵

    '''
    def __init__(self, sizeOfArena: float, number: int, speed: float, senseRadiu: float,noises: float, name: str = "vicsek" ) -> None:
        self.name = name
        print(name," is born")
        
        #####
        self.parameters_Init(sizeOfArena,number,speed,senseRadiu,noises) #init parameters
        

        pass
    
    def parameters_Init(self, sizeOfArena: float, number:int, speed: float, senseRadiu: float, noises: float) -> None:
        '''
        Init parameters

        Parameters
        --
        >>> sizeOfArena: float ( default: )
                the linear size of the squrae shape cell where simulations are carried out. 
        >>> number: int  ( default: )
                the number of units. 
        >>> speed: float ( default: )
                speed of the units(constant temporarily).
        >>> senseRadiu: float ( default: )
                the radius that an unit can feel.
        >>> noises: float ( default: )
                the strength of noises.

        '''
        self.l = sizeOfArena
        self.n = number
        self.v = speed
        self.r = senseRadiu
        self.yeta = noises
        # init Position P
        rng = np.random.default_rng()
        self.Po = rng.random((self.n,2))*self.l
        # init angle   Theta        0~2*pi but what is 
        self.Theta = rng.random((self.n,1))*2*np.pi 
        # init Velocity  V    
        self.Vo = np.hstack((self.v*np.cos(self.Theta),self.v*np.sin(self.Theta)))

        
        pass


#================== animation part

    def start(self) -> None:
        '''
        start the simulation
        
        Parameters:
        '''
        self.simulation_Init() # init file and folder 

        fig = plt.figure() # 
        plt.quiver(self.Po[:,0],self.Po[:,1],self.Vo[:,0],self.Vo[:,1])
        ani = animation.FuncAnimation(fig=fig, func=self._move, frames=self.stepNum-1, interval=20, blit=False, repeat=False) # frams-1是因为frame会传两个参数0
        plt.xlim((0, self.l))
        plt.ylim((0, self.l))
        ani.save('./vicsek.gif', fps=20)
        self.pbar.finish()
        print(":) \"Video is saved successfully.\"",self.name,"said")
        self.data_Save()
        print(":) \"Data is saved successfully.\"",self.name,"said")
        # plt.show()

    def simulation_Init(self):
        '''
        Init things which are used for simulation

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
        # self.stepNum = int(self.simulationTime/self.step)
        self.stepNum = 100
        self.now = 0 # record what the step is now, start from 0~(stepNum-1)
        
        # space init
        #----------------
        # space for data to be saved
        self.ThetaSaved = np.zeros((self.n, self.stepNum))

        # progressbar init
        #------------
        widgets = ['Progress: ',progressbar.Percentage(), ' ', progressbar.Bar('#'),' ', progressbar.Timer(),
           ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, max_value=self.stepNum).start()
        # self.pbar = progressbar.ProgressBar().start()

        #file init
        #-----------
        # init folder
        now = time.localtime()
        timeStr = time.strftime("%Y-%m-%d_%H-%M-%S",now)
        self.folderName = timeStr + '_' + str(self.n) + 'units_' + \
                    str(self.stepNum) + 'StepNumber_' + \
                    str(self.yeta) + 'Noise_' + \
                    str(self.l) + 'size_' + \
                    str(self.v) + 'speed'
        self.mypath = os.path.split(__file__)[0]
        os.chdir(self.mypath)
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
        stepNum = np.array([self.stepNum])
        self.file.create_dataset('angleSaved', data=self.ThetaSaved, compression='gzip', compression_opts=9)
        self.file.create_dataset('stepNum', data=stepNum, compression='gzip', compression_opts=9)
        self.file.close()
        os.chdir("..")
        pass



    def _move(self, frameNumber):
        '''
        # WARN!!!!!!!!!!
        do not touch easily
        由动画函数自动调用

        Parameters:
        ---
            frameNumber: 
                the number of the frame, 
        '''
        # print("\n___framsjlasjldjla",self.now)
        Po, Theta = self.update()
        self.ThetaSaved[:,self.now] = Theta.reshape((self.n, ))
        self.pbar.update(self.now+1)
        self.now +=1
        plt.cla()
        plt.xlim((0, self.l))
        plt.ylim((0, self.l))
        plt.quiver(Po[:,0],Po[:,1],self.Vo[:,0],self.Vo[:,1])
        pass


    #================ update data

    def update(self):
        '''
        # Main logic
        
        update the position.        ref:[2] Emergent Behavior in Flocks
        
        Parameters:

        '''
        dx = np.subtract.outer(self.Po[:, 0], self.Po[:, 0])
        dy = np.subtract.outer(self.Po[:, 1], self.Po[:, 1]) 
        distance = np.hypot(dx, dy)
        # periodic boundary
        Ax = (distance >= 0) * (distance <= self.r) # >=0是包括自己 
        #                                                     condition
        Ax += (dy > self.l/2) * (np.abs(dx)< self.l/2) * (np.hypot(0-dx,self.l-dy)<self.r)
        Ax += (dy > self.l/2) * (dx< -self.l/2)        * (np.hypot(-self.l-dx,self.l-dy)<self.r)
        Ax += (dy > self.l/2) * (dx> self.l/2)         * (np.hypot(self.l-dx,self.l-dy)<self.r)
        Ax += (dy < self.l/2) * (dx> self.l/2)         * (np.hypot(self.l-dx,0-dy)<self.r)#
        Ax += Ax.T
        # print(Ax)
        di = np.maximum(Ax.sum(axis=1), 1) #.reshape(self.n,1)
        Dx = np.diag(di)
        # print(Dx)
        Lx = Dx-Ax
        Id = np.identity(self.n)
        #  weight matrix
        #  Wx is a nonnegative asymmetric matrix whose wij element determines the interaction strength that particle i exerts on particle j
        Wx= np.ones((self.n,self.n))
        WxA = Ax * Wx

        # noise
        rng = np.random.default_rng()
        Noises = rng.random((self.n,1))*self.yeta - self.yeta/2
        
        
        
        self.Theta = np.arctan2(np.matmul(WxA,self.Vo[:,1].reshape(self.n,1)),\
                                np.matmul(WxA,self.Vo[:,0].reshape(self.n,1))+0.000000000001) 
        self.Theta = self.Theta + Noises

        self.Theta = np.mod(self.Theta, 2 * np.pi)
        '''
        ps: “.” 加上运算符表示按元素进行运算
        Theta(t+1) =  <Theta(t)> + noise              
                                (Wx.*Ax)*Vy(t)                    
        <Theta(t)> = arctan(  ————————————————————— )         Vx为速度x的分量, Vy同理, epsilon趋于0(防止分母为零)
                            (Wx.*Ax)*Vx(t)+epsilon
        
        '''
        # print(self.Theta)
        # speed remains unchanged
        self.Vo = np.hstack((self.v*np.cos(self.Theta),self.v*np.sin(self.Theta)))
        # print(self.Vo)
        self.Po  = self.Po + self.Vo * 1 # 1 means time
        # print(self.P) 
        self.Po = np.mod(self.Po, self.l) # 取余
        return self.Po, self.Theta


#%%
if __name__ == "__main__":
    #使用类和对象的方式，这样可以同时跑多个参数
    vicsek = Vicsek(sizeOfArena= 100.0, number= 200, speed=.5, senseRadiu=3, noises=.05,name="vicsek_A")
    vicsek2 = Vicsek(sizeOfArena= 100.0, number= 400, speed=.5, senseRadiu=3, noises=.05,name="vicsek_AA")
    vicsek.start()
    ##########
    vicsek2.start()



'''
[1] Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I. & Shochet, O. Novel Type of Phase Transition in a System of Self-Driven Particles. Phys. Rev. Lett. 75, 1226-1229 (1995).
[2] Cucker, F. & Smale, S. Emergent Behavior in Flocks. IEEE Trans. Automat. Contr. 52, 852-862 (2007).
[3] Sattari, S. et al. Modes of information flow in collective cohesion. SCIENCE ADVANCES 14 (2022).


'''
