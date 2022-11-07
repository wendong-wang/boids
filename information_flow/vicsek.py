# vicsek model simulation
# for testing 
# authority: Pang Jiahun
# start time: 2022/11/7
# last time: 2022/11/7
# end time: ~

'''
version description:
    角度0~2pi，直接取平均，参照[2]中的original vicsek
'''
#考虑：
#   1. 动画部分要不要放入VICSEK类中
#   2. 加入惯性，而不是角度取平均
#   3. 数学计算采用[2]的方式？还是逐个元素进行相应的计算
#   4. 还没有考虑数据存储的部分
#   5.角度取平均，还有个问题，那便是，0与2pi该怎么 处理-》有没有一个循环函数（所以原文用了sin，cos来取平均，而不是角度直接取平均）


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
    # v: velocities of the units ( temporarily) 1x1 速度大小
    # Theta: directions nx1 角度(每个unit运动的不同)
    # P: matrix of the positions of the units nx2 位置矩阵
    # r: the radius that an unit can feel 1x1 所感知的半径
    # yeta: the strength of the noises 1x1 噪声
    大小写规范问题:小写为数值，大写为矩阵
    '''
    def __init__(self,sizeOfArena: float, number: int, speed: float, senseRadiu: float,noises: float) -> None:
        self.parameters_Init(sizeOfArena,number,speed,senseRadiu,noises) #init parameters
        print("hello world")
        self.simulationTime = 10 # the duration of the simulation, unit: second
        self.step = 0.01 # the duration of the step
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
        self.Theta = rng.random((self.n,1))*2*np.pi
        



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
    ax = fig.subplots()
    process, = ax.plot(vicsek.P[:,0],vicsek.P[:,1],c='k', marker ='o', ls='None')
    print(type(process))
    ani = animation.FuncAnimation(fig=fig, func=_move, fargs= (vicsek,process, ), frames=100, interval=20, blit=False)
    plt.xlim((0, vicsek.l))
    plt.ylim((0, vicsek.l))
    plt.show()

def _move(frameNumber, vicsek: VICSEK, process):
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
        process:
            封装后的数据
    '''

    P = update(vicsek=vicsek)
    process.set_ydata(P[:,1])
    process.set_xdata(P[:,0])
    return process



#================ update data
#%%
def update(vicsek: VICSEK):
    '''
    update the position.        ref:[2] Emergent Behavior in Flocks->角度取平均问题

    Parameters:
        vicsek: VICSEK(class) default: 
            vicsek model
    '''
    dx = np.subtract.outer(vicsek.P[:, 0], vicsek.P[:, 0])
    dy = np.subtract.outer(vicsek.P[:, 1], vicsek.P[:, 1]) 
    distance = np.hypot(dx, dy)
    Ax = (distance >= 0) * (distance < vicsek.r)# >=0是包括自己
    di = np.maximum(Ax.sum(axis=1), 1)#是包括自己
    Dx = np.diag(di)
    Lx = Dx-Ax
    id = np.identity(vicsek.n)
    # print(vicsek.Theta)
    #noise
    rng = np.random.default_rng()
    Noises = rng.random((vicsek.n,1))*vicsek.yeta -vicsek.yeta/2
    vicsek.Theta = np.matmul((id - np.matmul(np.linalg.inv(Dx),Lx)),vicsek.Theta)
    vicsek.Theta += Noises
    vicsek.P = vicsek.P + np.hstack((vicsek.v*np.cos(vicsek.Theta),vicsek.v*np.sin(vicsek.Theta)))
    # vicsek.P = vicsek.P % vicsek.l#取余
    
    vicsek.P = np.mod(vicsek.P, vicsek.l)#取余
    return vicsek.P


#%%
if __name__ == "__main__":
    #修改参数就修改类的参数就行，这样可以同时跑多个参数
    vicsek = VICSEK(sizeOfArena= 100.0, number= 500, speed=1.0, senseRadiu= 5, noises=.5)
    animate(vicsek)



'''
[1] Novel Type of Phase Transition in a System of Self-Driven Particles
[2] Emergent Behavior in Flocks
'''

