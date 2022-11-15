#data processing 
# Under developing
# authority: Pang Jiahuan
# start time: 2022/10/30
# last time: 2022/11/11
# end time: ~
#information flow: mutual information; 
#                  time-delayed mutual information; 
#                  transfer entropy; 
#                  shared, intrinsic, syn.. entropy.

'''
version description:
    读取数据,将H5PY常用操作包装成一个class
    ！！！现在这个文件还不可以运行
'''
#考虑：
#   1. 信息流处理类的建立

import h5py
import numpy as np
import os

class H5PY_PROCESSOR():
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
        Using depth-first algorithm to draw the file structure of HDF5 group. 

        Parameters
        -------
        path: str,
            the absolute path of a group or a database
        name: str, default: '/'
            the name of a group or a database, and the default value poits to root directory.
        
        Warning
        ----------
        this function now is under development and is not able to vetify the correctness of the input parameters. 
        
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

class INFORMATION_PROCESSOR():
    def __init__(self, Theta) -> None:
        self.Theta = Theta
        pass
    def mutual_Information(self):
        pass
    def time_Delayed_Mutual_Information(self):
        pass
    def transfer_Entropy(self):
        pass
    def intrinsic_Information_Flow(self):
        pass
    def shared_Information_Flow(self):
        pass
    def synergistic_Information_Flow(self):
        pass



if __name__ == "__main__":
    mypath = os.path.split(__file__)[0]
    os.chdir(mypath)
    #
    folderName = "2022-11-11_22-28-47_400units_100StepNumber_0.05Noise_100.0size_0.5speed" #想要处理的数据名称
    os.chdir(folderName)
    # f = H5PY_PROCESSOR("h5py_example.hdf5",'r')
    # f.search_Deep("/")
    # f.close()