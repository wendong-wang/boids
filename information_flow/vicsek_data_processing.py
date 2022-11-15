#data processing 
# Under developing
# authority: Pang Jiahuan
# start time: 2022/10/30
# last time: 2022/11/15
# end time: ~
#information flow: mutual information; 
#                  time-delayed mutual information; 
#                  transfer entropy; 
#                  shared, intrinsic, syn.. entropy.
# python: 3.6
'''
version description:
    初步实现所要熵的计算,intrinsic information flow的正确性还有待考究
'''
#考虑：
#   ·. 取和不为1
#   ·.intrinsic information flow
import h5py
import numpy as np
import os

import itertools as it #排列组合用
import dit      

class H5PY_Processor():
    '''
    A Class for basic HDF5 file operation

    Init Parameters
    --
    filename: str
        name of the file

    authority: str
        the authority of operation:
        - r  只读，文件必须已存在 read-only;
        - r+ 读写，文件必须已存在 read and write;
        - w  新建文件，若存在覆盖;
        - w- 或x, 新建文件, 若存在报错;
        - a  如存在则读写，不存在则创建(默认).
    '''
    
    def __init__(self, fileName: str, authority: str) -> None:
        self.f = h5py.File(fileName, authority)
        pass
    def close(self):
        print("文件被释放")
        self.f.close()
        pass
    def search_Deep(self,path: str,name:str = '/') -> None:
        '''
        Use depth-first algorithm to draw the file structure of HDF5 groups. 

        Parameters
        -------
        path: str,
            the absolute path of a group or a database
        name: str, default: '/'
            the name of a group or a database, and the default value poits to root directory.
        
        Warning
        ----------
        this function now is under development and is not able to vertify the correctness of the input parameters.

        Examples:
        ---
            self.search_Deep(path = '/')
        
        '''
        dog = self.f[path] # a group or dataset
        if dog.name == "/":
            print('+-',end='')
            count = 0
        else:
            count = dog.name.count("/")
        for i in range(count):
            print("+----",end='')
        #Group有.keys()方法 而dataset没有
        #dataset有.len()方法

        try:# if is a group
            length = len(dog.keys())
            print(name,"[G]")
            if length != 0:
                for key in dog.keys():
                    self.search_Deep(dog.name +"/"+key,key)
        except:# a dataset
            # print(name,"[D]", np.shape(dog),end = '')#没有长度的dataset可能会被看作是group，因此len()失效
            # print(dog)
            print(name,"[D]", np.shape(dog))#没有长度的dataset可能会被看作是group，因此len()失效

class Information_Processor():
    def __init__(self, Theta, stepNum) -> None:
        self.Theta = Theta
        self.stepNum = stepNum
        self.discretize()
        #init alphabet
        self.alphabet={}
        for e in it.product('012345', repeat=3):
            a = ''.join(e)
            self.alphabet[a] = 0.0   
        # print(self.alphabet)  
        self.count_Distribution(x=0,y=1)
        self.calculate_Information()
        pass

    def discretize(self):
        '''
        discretize the angle set (self.Theta (0~2*pi )) into 6 parts
        0 1 2 3 4 5  

        for example: [0°,60°) -> 0; [60°,120°) -> 1 ...

        '''
        Theta = self.Theta
        Theta = np.mod(Theta,2*np.pi)#将2*pi变为0
        Theta = np.floor_divide(Theta, np.pi/3)
        self.bins = Theta 
        pass


    def count_Distribution(self, x:int, y:int):
        '''
        count the occurrences of  (x(t), y(t), y(t+τ)) 
            x affect y. For convenience, let X, Y, and Z denote x(t), y(t), y(t+τ)seperately.

        Parameters
        --
        >>> x,y: int
            the index of vairable

        
        '''
        # count
        for i in range(self.stepNum-1):
            index = str(int(self.bins[i][x])) + str(int(self.bins[i][y])) + str(int(self.bins[i+1][y]))
            # print(index)
            self.alphabet[index] += 1/(self.stepNum-1)

        alphabetKeys  = list(self.alphabet.keys()) # seperate the keys form dict
        alphabetValue = list(self.alphabet.values()) # seperate the value from dict

        self.XYZ = dit.Distribution(alphabetKeys, alphabetValue)
        self.XYZ.set_rv_names('XYZ')
        # print(self.XYZ)
        pass



        

    def mutual_Information(self):
        '''
        calculate the mutual information of x(t) and y(t)

        Return
        ---
            mI: mutual information of x(t) and y(t)
        '''
        
        mI = dit.shannon.mutual_information(self.XYZ,'X','Y')
        # print(mutual_Information)
        return mI
        
    def time_Delayed_Mutual_Information(self):
        '''
        calculate time delayed mutual information (TDMI)
        
        Return
        ---
            tMDI: TMDI
        '''
        tMDI = dit.shannon.mutual_information(self.XYZ,'X','Z')
        return tMDI

    def transfer_Entropy(self):
        '''
        calculate  transfer_Entropy (TE)
        
        Return
        ---
            tE: TE
        '''
        # tE = dit.multivariate.coinformation(self.XYZ, rvs = 'XZ', crvs = 'Y')
        tE = dit.multivariate.coinformation(self.XYZ, 'XZ','Y')
        return tE


    def intrinsic_Information_Flow(self):
        '''
        calculate time intrinsic_Information_Flow (IIF)
        
        Return
        ---
            iIF: IIF
        '''
        iIF = dit.multivariate.secret_key_agreement.intrinsic_mutual_information(self.XYZ, 'XZ','Y')
        return iIF

    def shared_Information_Flow(self, tE, iIF):
        '''
        calculate time shared_Information_Flow (SHIF)
        
        Parameters
        ---
        >>> tE: TE
        >>> iIF: IIF

        Return
        ---
            sHIF: SHIF
        '''
        sHIF = tE - iIF
        return sHIF



    def synergistic_Information_Flow(self, tDMI, iIF):
        '''
        calculate time synergistic_Information_Flow (SYIF)
        
        Parameters
        ---
        >>> tDMI: TDMI
        >>> iIF: IIF

        Return
        ---
            sYIF: SYIF
        '''
        sYIF = tDMI - iIF
        return sYIF

    def calculate_Information(self):
        '''
        calculate all the information we need
        '''
        self.mI = self.mutual_Information()
        self.tDMI = self.time_Delayed_Mutual_Information()
        self.tE = self.transfer_Entropy()
        self.iIF = self.intrinsic_Information_Flow()
        self.sHIF = self.shared_Information_Flow(tE= self.tE, iIF= self.iIF)
        self.sYIF = self.synergistic_Information_Flow(tDMI=self.tDMI, iIF=self.iIF)
        pass

if __name__ == "__main__":
    # enter the absolute path of this python file
    mypath = os.path.split(__file__)[0]
    os.chdir(mypath)

    # enter the folder containing the data
    folderName = "2022-11-14_22-11-36_200units_100StepNumber_0.05Noise_100.0size_0.5speed" #想要处理的所在数据名称
    os.chdir(folderName)
    # get the file
    f = H5PY_Processor("vicsekData.hdf5","r")
    # f.search_Deep("/")
    processor = Information_Processor(Theta= f.f["angleSaved"][:,:],stepNum=f.f['stepNum'][0])
    f.close()

    print("multual_Information:",processor.mI)
    print("time_Delayed_Mutual_Information:",processor.tDMI)
    print("transfer_Entropy:",processor.tE)
    print("intrinsic_Information_Flow:",processor.iIF)
    print("shared_Information_Flow:",processor.sHIF)
    print("synergistic_Information_Flow:",processor.sYIF)

'''
reference:
1. G. James, R., J. Ellison, C. & P. Crutchfield, J. dit: a Python package for discrete information theory. JOSS 3, 738 (2018).
'''