import h5py
import numpy as np
import os

import itertools as it # sorting
import dit # information theory
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr #person correlation 
import time


import sys 
sys.path.append(".") 
from information_processor import Information_Processor
from h5py_process import H5PY_Processor
'''
#Attention !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

文件排序方式为:名称+增序

'''



if __name__ == "__main__":
    # enter the absolute path of this python filep
    mypath = os.path.split(__file__)[0]
    os.chdir(mypath)
    os.chdir("..")
    
    # os.chdir("..")
    # os.chdir("Intrinsic_mutual_information")

    #找到路径
    os.chdir("20240717")
    # os.chdir("Data_Vicsek_for_v_a_changeEta_50")
    os.chdir("Data_Vicsek_wlf5_suplymentary")
    # os.chdir("Data_Vicsek_loseMemory_onlyLeaderAffect")
    

    

    noThroughList = ["done", "v_a result", "v_a result_1"]
    for folder in os.listdir("."): # folder is the name of each folder
        if folder in noThroughList or folder[-5:] ==".xlsx" or folder[-4:]==".txt" or folder[-5:]==".opju":
            continue
        print(folder)
        os.chdir(folder)


    
        f = H5PY_Processor("vicsekData.hdf5","r")
        Ve             = f.f["velocitySaved"][:,:,:]
        Theta          = f.f["angleSaved"][:,:]
        adjacentMatrix    = f.f["adjacentSaved"][:,:,:]
        interactionMatrix = f.f["interactionSaved"][:,:]
        # positionSaved = f.f["positionSaved"][:,:]
        # angleInformation    = f.f["angleInformationSaved"][:,:,:]

        groupParameter =f.f["parameters"]
        timeResolution = groupParameter["timeResolution"][:][0]
        numOfSteps        = groupParameter["numOfSteps"][:][0]
        # numOfSteps        = groupParameter["stepNum"][:][0]
        wLF            = groupParameter["wLF"][:][:]
        numOfLeaders      = groupParameter["numOfLeaders"][:][0]
        # numOfLeaders      = groupParameter["leaderNum"][:][0]
        noiseStrength  = groupParameter["noiseStrength"][:][0]
        numOfNodes       = groupParameter["numOfNodes"][:][0]
        # numOfNodes       = groupParameter["nodesNum"][:][0]
        # binsNum       = groupParameter["binsNum"][:][0]

        f.close()
        # exit()

        followerList = np.arange(numOfLeaders,numOfNodes)

        x = 0# init
        y = 0
        z = 0
        count = 0
        maxColorbar=0
        minColorbar=1000

        binsNum = 8

        inf = Information_Processor(Theta = Theta, numOfSteps = numOfSteps, x=x, y=y,z=z, binsNum=binsNum)#leaer to follower#!!!!!!!!!!!!!!!!
        time_start = time.time()  # 记录开始时间
        # function()   执行的程序
        inf.discretize()
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print("discretize",time_sum)
        #计算
        informationList = {
                        "mutual_information": np.zeros((numOfNodes,numOfNodes)),
                        "time_delayed_mutual_information": np.zeros((numOfNodes,numOfNodes)),
                        "transfer_entropy":np.zeros((numOfNodes,numOfNodes)),
                        "intrinsic_information_flow":np.zeros((numOfNodes,numOfNodes)),
                        "shared_information_flow":np.zeros((numOfNodes,numOfNodes)),
                        "synergistic_information_flow":np.zeros((numOfNodes,numOfNodes)),
                    }
        informationListCompare = {
                        "mutual_information": np.zeros((numOfNodes,numOfNodes)),
                        "time_delayed_mutual_information": np.zeros((numOfNodes,numOfNodes)),
                        "transfer_entropy":np.zeros((numOfNodes,numOfNodes)),
                        "intrinsic_information_flow":np.zeros((numOfNodes,numOfNodes)),
                        "shared_information_flow":np.zeros((numOfNodes,numOfNodes)),
                        "synergistic_information_flow":np.zeros((numOfNodes,numOfNodes)),
                    }               
        #计算
        time_startall = time.time()  # 记录开始时间
        
        # only calculate the information leader gives to follower 
        for i in range(numOfLeaders):
            for j in range(numOfNodes):
        # # for i in range(numOfNodes):
        # #     for j in range(numOfNodes):
        # for i in range(10):
        #     for j in range(10):
                if(i==j):
                    print("到",i,j,"了")
                    # continue
                inf.x = j# X_t

                inf.y = i# Y_t
                # inf.y = (i+1)%2# Y_t

                inf.z = j # Z_t Y_t+1

                # 计算分布
                time_start = time.time()  # 记录开始时间


                ## 计算邻居的
                resultFlag = inf.count_Distribution_Adjacent(adjacentMatrix=adjacentMatrix)
                # 如果数据太少
                if resultFlag == False:
                    print("%d与%d做邻居的时间太短",x,y)
                    continue
                
                # resultFlag = inf.count_Distribution(x=inf.x,y=inf.y, z=inf.z)
                # function()   执行的程序

                time_end = time.time()  # 记录结束时间
                time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                print("count",time_sum)


                # 计算信息量
                time_start = time.time()  # 记录开始时间
                inf.calculate_Information()
                time_end = time.time()  # 记录结束时间

                time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                print("calculate",time_sum)
                informationDictTemp=inf.get_Information()
                for keys in informationDictTemp:
                    # print(np.shape(informationList[keys][:,:]))
                    informationList[keys][i][j] = informationDictTemp[keys]
                    if informationDictTemp[keys]>maxColorbar:
                        maxColorbar = informationDictTemp[keys]
                    if informationDictTemp[keys]<minColorbar:
                        minColorbar = informationDictTemp[keys]

        # function()   执行的程序
        time_endall = time.time()  # 记录结束时间
        time_sumall = time_endall - time_startall  # 计算的时间差为程序的执行时间，单位为秒/s
        print("all:",time_sumall)
        
        #二值比较
        for i in range(numOfNodes):
            for j in range(numOfNodes):
                if(i>=j):
                    continue
                for keys in informationListCompare:
                    if(informationList[keys][i][j]-informationList[keys][j][i]>0):
                        if informationList[keys][j][i]>0:    
                            informationListCompare[keys][i][j]=informationList[keys][i][j]/informationList[keys][j][i]
                            informationListCompare[keys][j][i]=informationList[keys][j][i]/informationList[keys][i][j]
                        else:
                            informationListCompare[keys][i][j]=1
                            informationListCompare[keys][j][i]=0
                    elif(informationList[keys][i][j]-informationList[keys][j][i]<0):
                        if informationList[keys][j][i]>0:    
                            informationListCompare[keys][i][j]=informationList[keys][i][j]/informationList[keys][j][i]
                            informationListCompare[keys][j][i]=informationList[keys][j][i]/informationList[keys][i][j]
                        else:
                            informationListCompare[keys][i][j]=0
                            informationListCompare[keys][j][i]=1
        
        

        if not os.path.isdir("NON"):
            os.mkdir("NON")
        os.chdir("NON") 



        # file = h5py.File("informationHeatmap_all_"+str(binsNum)+".hdf5", 'w-')
        file = h5py.File("IMI_YZ_nei_"+str(binsNum)+".hdf5", 'w-')
        # file = h5py.File("IMI_XZ_nei_"+str(binsNum)+".hdf5", 'w-')
        # file = h5py.File("IMI_XZcom_"+str(binsNum)+".hdf5", 'w-')
        # file = h5py.File("IMI_XZcom_nei_"+str(binsNum)+".hdf5", 'w-')
        # file = h5py.File("IMI_XYZ_nei_"+str(binsNum)+".hdf5", 'w-')
        # file = h5py.File("IMI_nei_cor"+str(binsNum)+".hdf5", 'w-')
        numOfNodes = np.array([numOfNodes])
        numOfLeaders = np.array([numOfLeaders])  
        wLF = np.array([wLF])
        noiseStrength = np.array([noiseStrength])
        # angleInfor = np.array(angleInformation)
        print(numOfNodes)
        file.create_dataset("nodesNum",    data=numOfNodes,           compression='gzip',   compression_opts=9)
        file.create_dataset("numOfLeaders",    data=numOfLeaders,           compression='gzip',   compression_opts=9)
        file.create_dataset("wLF",    data=wLF,           compression='gzip',   compression_opts=9)
        file.create_dataset("noiseStrength",    data=noiseStrength,           compression='gzip',   compression_opts=9)
        file.create_dataset("adjacentMatrix",    data=adjacentMatrix,           compression='gzip',   compression_opts=9)
        for key in informationList:
            keyB = key[:5] + "Com"#"Binary"
            file.create_dataset(key, data=informationList[key], compression='gzip', compression_opts=9)
            file.create_dataset(keyB, data=informationListCompare[key], compression='gzip', compression_opts=9)
            # file.create_dataset('angleInfor', data=angleInformation, compression='gzip', compression_opts=9)
        file.close()
        os.chdir("..")
        os.chdir("..")
