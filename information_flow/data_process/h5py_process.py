import h5py
import numpy as np


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
        # print("文件被释放")
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
        >>> self.search_Deep(path = '/')
        
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