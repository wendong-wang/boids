# vicsek model simulation
# for testing 
# authority: Pang Jiahun
# start time: 2022/11/7
# last time: 2022/11/8
# end time: ~

'''
version description:
    考虑速度连续,参照[2]的表示方法，现在的编程思路是一种中和状态，既有Theta变量又有速度

相对与上一版改进的说明：
    加入惯性,周期边界条件,画图变成了箭头
'''
#考虑：
#   1. 动画部分要不要放入VICSEK类中
#   2. 数学计算采用[2]的方式？还是逐个元素进行相应的计算
#   3. 还没有考虑数据存储的部分
#   4. 取平均改用速度的方式，角度-》当趋向一致的时候老是抖，无法过度


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
    Ax = (distance >= 0) * (distance < vicsek.r)  # >=0是包括自己 
    # Ax += (np.abs(dx) >= np.abs(dy)) * (vicsek.l* distance/np.abs(dx+0.000001) - distance)<vicsek.r # 左右
    # Ax += (np.abs(dx) < np.abs(dy)) *(vicsek.l* distance/np.abs(dy+0.000001) - distance)<vicsek.r # 上下
    Ax += (dy > vicsek.l/2) * (np.abs(dx)< vicsek.l/2) * (np.hypot(0-dx,vicsek.l-dy)<vicsek.r)
    Ax += (dy > vicsek.l/2) * (dx< -vicsek.l/2) * (np.hypot(-vicsek.l-dx,vicsek.l-dy)<vicsek.r)
    Ax += (dy > vicsek.l/2) * (dx> vicsek.l/2) * (np.hypot(vicsek.l-dx,vicsek.l-dy)<vicsek.r)
    Ax += (dy < vicsek.l/2) * (dx> vicsek.l/2) * (np.hypot(vicsek.l-dx,0-dy)<vicsek.r)#
    Ax += Ax.T

    di = np.maximum(Ax.sum(axis=1), 1)#.reshape(vicsek.n,1)#是包括自己
    Dx = np.diag(di)
    Lx = Dx-Ax
    id = np.identity(vicsek.n)
    # print(di)
    # print(di.reshape(vicsek.n,1))
    #noise
    rng = np.random.default_rng()
    Noises = rng.random((vicsek.n,1))*vicsek.yeta -vicsek.yeta/2
    
    # vicsek.Theta = np.matmul((id - np.matmul(np.linalg.inv(Dx),Lx)),vicsek.Theta)
    K = 0.1
    Z = np.arctan2((np.matmul(Ax,vicsek.V[:,1]).reshape(vicsek.n,1)/di.reshape(vicsek.n,1)-vicsek.v*np.sin(vicsek.Theta)),((np.matmul(Ax,vicsek.V[:,0]).reshape(vicsek.n,1)/di.reshape(vicsek.n,1)-vicsek.v*np.cos(vicsek.Theta)))) 
    # vicsek.Theta += K*np.arctan((np.matmul(Ax,vicsek.V[:,1]).reshape(vicsek.n,1)/di.reshape(vicsek.n,1)-vicsek.v*np.sin(vicsek.Theta))/((np.matmul(Ax,vicsek.V[:,0]).reshape(vicsek.n,1)/di.reshape(vicsek.n,1)-vicsek.v*np.cos(vicsek.Theta))+0.00000001)) * vicsek.step
    vicsek.Theta += K*Z* vicsek.step
    '''
    上面的计算变成代码确实有点复杂，但公式还算简单.很多代码是调整矩阵的格式用的
    
    Theta(t+1) = Theta(t) +  K .* d(Theta) .* dt                      ps: “.” 加上运算符表示按元素进行运算, K是一个常量
                         Dx^{-1}*Ax*Vy(t)-v.*sin(Theta(t))                    
    d(Theta) = arctan(  —————————————————————————————————————— )         Vx为速度x的分量, Vy同理, epsilon趋于0(防止分母为零)
                         Dx^{-1}*Ax*Vx(t)-v.*cos(Theta(t))+epsilon
    
    因此速度变化就是连续了，不会出现方向太大的突变
    方向差不多的时候就会抖动,由于角度范围并不是连续的原因，并且无法从-pi相互过渡到pi，所以打算不从不从角度上下手了
    '''
    # print(Z[1]/np.pi*180/vicsek.step/K)
    vicsek.Theta += Noises
    vicsek.V = np.hstack((vicsek.v*np.cos(vicsek.Theta),vicsek.v*np.sin(vicsek.Theta)))
    vicsek.P = vicsek.P + np.hstack((vicsek.v*np.cos(vicsek.Theta),vicsek.v*np.sin(vicsek.Theta)))*vicsek.step
    # vicsek.P = vicsek.P % vicsek.l#取余
    
    vicsek.P = np.mod(vicsek.P, vicsek.l)#取余
    return vicsek.P


#%%
if __name__ == "__main__":
    #修改参数就修改类的参数就行，这样可以同时跑多个参数
    vicsek = VICSEK(sizeOfArena= 80.0, number= 200, speed=5, senseRadiu= 10, noises=0.5)
    animate(vicsek)



'''
[1] Novel Type of Phase Transition in a System of Self-Driven Particles
[2] Emergent Behavior in Flocks
'''
