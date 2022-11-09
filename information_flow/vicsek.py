# vicsek model simulation
# for testing 
# authority: Pang Jiahuan
# start time: 2022/11/7
# last time: 2022/11/9
# end time: ~

'''
version description:
    考虑速度连续,参照[2]的表示方法，现在的编程思路是一种中和状态，既有Theta变量又有速度

相对与上一版改进的说明：
    取消了惯性，改用速度取平均，放弃arctan函数
'''
#考虑：
#   1. 动画部分要不要放入VICSEK类中
#   2. 数学计算采用[2]的方式？还是逐个元素进行相应的计算
#   3. 还没有考虑数据存储的部分
#   4. 加速度的表示，注意量纲dimension

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VICSEK():
    '''
    class of Vicsek model, each class is a group of Vicsek model

    Parameters
    --
        sizeOfArena: float ( default: )
            the linear size of the squrae shape cell where simulations are carried out. 
        number: int  ( default: )
            the number of units. 
        speed: float ( default: )
            speed of the units(constant temporarily).
        senseRadiu: float ( default: )
            the radius that an unit can feel.
        noises: float ( default: )
            the strength of noises
    
    > hello

    '''
    '''
    Notation
    --
    # l: the simulations are carried out in a square shape cell of linear size L 1x1 场地大小
    # n: n units 1x1 个数
    # v: value of the velocities of the units  1x1 速度大小
    # Theta: directions nx1 角度(每个unit运动的不同)
    # V: velocities of the units nx2 
    # P: matrix of the positions of the units nx2 位置矩阵
    # r: the radius that an unit can feel 1x1 所感知的半径
    # yeta: the strength of the noises 1x1 噪声
    大小写规范问题:小写为数值，大写为矩阵
    '''
    def __init__(self,sizeOfArena: float, number: int, speed: float, senseRadiu: float,noises: float) -> None:
        self.parameters_Init(sizeOfArena,number,speed,senseRadiu,noises) #init parameters
        print("hello world")
        self.simulationTime = 10 # the duration of the simulation, unit: second
        self.step = 0.1 # the duration of the step
        pass
    
    def parameters_Init(self, sizeOfArena: float, number:int, speed: float, senseRadiu: float, noises: float) -> None:
        '''
        Init parameters

        Parameters
        --
            sizeOfArena: float ( default: )
                the linear size of the squrae shape cell where simulations are carried out. 
            number: int  ( default: )
                the number of units. 
            speed: float ( default: )
                speed of the units(constant temporarily).
            senseRadiu: float ( default: )
                the radius that an unit can feel.
            noises: float ( default: )
                the strength of noises
        '''
        self.l = sizeOfArena
        self.n = number
        self.v = speed
        self.r = senseRadiu
        self.yeta = noises
        # init P and theta
        rng = np.random.default_rng()
        self.P = rng.random((self.n,2))*self.l
        # init Theta
        self.Theta = rng.random((self.n,1))*2*np.pi-np.pi
        # init Velocity
        self.V = np.hstack((self.v*np.cos(self.Theta),self.v*np.sin(self.Theta)))
        



#================== animation part
#%%
def animate(vicsek) -> None:
    '''
    start the simulation
    
    Parameters:
        vicsek: VICSEK(class) default: 
            vicsek model

    '''
    # print(vicsek)
    # print(id(vicsek))
    fig = plt.figure()
    # ax = fig.subplots()
    # process, = ax.plot(vicsek.P[:,0],vicsek.P[:,1], c='k', marker ='.', ls='None')
    plt.quiver(vicsek.P[:,0],vicsek.P[:,1],vicsek.V[:,0],vicsek.V[:,1])

    ani = animation.FuncAnimation(fig=fig, func=_move, fargs= (vicsek, ), frames=100, interval=20, blit=False)
    plt.xlim((0, vicsek.l))
    plt.ylim((0, vicsek.l))
    plt.show()

def _move(frameNumber, vicsek: VICSEK):
    '''
    # WARN!!!!!!!!!!
    do not touch easily
    由动画函数自动调用

    Parameters:
    ---
        frameNumber: 
            the number of the frame, 
        vicsek: VICSEK(class) default: 
            vicsek model
    '''

    P = update(vicsek=vicsek)
    plt.cla()
    plt.xlim((0, vicsek.l))
    plt.ylim((0, vicsek.l))
    plt.quiver(P[:,0],P[:,1],vicsek.V[:,0],vicsek.V[:,1])
    # return process
    pass



#================ update data
#%%
def update(vicsek: VICSEK):
    '''
    update the position.        ref:[2] Emergent Behavior in Flocks

    Parameters:
        vicsek: VICSEK(class) default: 
            vicsek model
    '''
    dx = np.subtract.outer(vicsek.P[:, 0], vicsek.P[:, 0])
    dy = np.subtract.outer(vicsek.P[:, 1], vicsek.P[:, 1]) 
    distance = np.hypot(dx, dy)
    # periodic boundary
    Ax = (distance >= 0) * (distance <= vicsek.r) # >=0是包括自己 

    Ax += (dy > vicsek.l/2) * (np.abs(dx)< vicsek.l/2) * (np.hypot(0-dx,vicsek.l-dy)<vicsek.r)
    Ax += (dy > vicsek.l/2) * (dx< -vicsek.l/2) * (np.hypot(-vicsek.l-dx,vicsek.l-dy)<vicsek.r)
    Ax += (dy > vicsek.l/2) * (dx> vicsek.l/2) * (np.hypot(vicsek.l-dx,vicsek.l-dy)<vicsek.r)
    Ax += (dy < vicsek.l/2) * (dx> vicsek.l/2) * (np.hypot(vicsek.l-dx,0-dy)<vicsek.r)#
    Ax += Ax.T

    di = np.maximum(Ax.sum(axis=1), 1) #.reshape(vicsek.n,1)
    Dx = np.diag(di)
    
    Lx = Dx-Ax
    Id = np.identity(vicsek.n)
    # noise
    rng = np.random.default_rng()
    Noises = rng.random((vicsek.n,1))*vicsek.yeta - vicsek.yeta/2
    
    vicsek.V = np.matmul((Id - np.matmul(np.linalg.inv(Dx),Lx)),vicsek.V)
    
    '''
    上面的计算变成代码确实有点复杂，但公式还算简单.很多代码是调整矩阵的格式用的
    ps: “.” 加上运算符表示按元素进行运算
    V(t+1) = V(t) - Dx^{-1} * Lx * V(t) = (Id - Dx^{-1}* Lx) * V(t)
    
    '''

    vicsek.V += Noises
    
    # speed remains unchanged
    velocityValue = np.hypot(vicsek.V[:,0],vicsek.V[:,1])
    vicsek.V[:,0] *= vicsek.v/velocityValue
    vicsek.V[:,1] *= vicsek.v/velocityValue
    vicsek.Theta = np.arccos(vicsek.V[:,0]/velocityValue)

    vicsek.P  = vicsek.P + vicsek.V * vicsek.step 
    # print(vicsek.P)
    # vicsek.V = np.hstack((vicsek.v*np.cos(vicsek.Theta),vicsek.v*np.sin(vicsek.Theta)))
    # vicsek.P = vicsek.P + np.hstack((vicsek.v*np.cos(vicsek.Theta),vicsek.v*np.sin(vicsek.Theta)))*vicsek.step
    # vicsek.P = vicsek.P % vicsek.l#取余
    
    vicsek.P = np.mod(vicsek.P, vicsek.l) # 取余
    return vicsek.P


#%%
if __name__ == "__main__":
    #修改参数就修改类的参数就行，这样可以同时跑多个参数
    vicsek = VICSEK(sizeOfArena= 200.0, number= 400, speed=10, senseRadiu=5, noises=0.6)
    animate(vicsek)



'''
[1] Novel Type of Phase Transition in a System of Self-Driven Particles
[2] Emergent Behavior in Flocks
'''
