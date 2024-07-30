import numpy as np
import itertools as it # sorting
import dit # information theory





class Information_Processor():
    def __init__(self, Theta, numOfSteps, x, y, z, binsNum) -> None:
        '''
        x: X, y:Y, z:Z. We are calculating the effect X,Y gives to Z (X gives to Z and exclude the influence of Y)
        '''
        self.Theta = Theta
        self.numOfSteps = numOfSteps
        self.x = x
        self.y = y
        self.z = z
        self.binsNum = binsNum
        # self.discretize()
        self.alphabetBook = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g',\
                              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w'\
                                'x', 'y', 'z']
        #init alphabet
        self.alphabet= self.want_An_Empty_Distribution()
        



    def discretize(self):
        '''
        discretize the angle set (self.Theta (0~2*pi )) into ''binsNum'' parts
        0 1 2 3 4 5  

        for example: [0°,60°) -> 0; [60°,120°) -> 1 ...

        '''
        Theta = np.round(self.Theta*180/np.pi,4)
        Theta = np.mod(Theta,360)#将2*pi变为0
        # Theta = np.floor_divide(Theta, 2*np.pi/self.binsNum)
        Theta = np.floor_divide(Theta, 360/self.binsNum)
        self.bins = Theta 
        
        pass

    def want_An_Empty_Distribution(self, number = 3):
        alphabetSequence = ''
        for i in range(self.binsNum):
            alphabetSequence += self.alphabetBook[i] 
        alphabet={}
        for e in it.product(alphabetSequence, repeat=number):
            a = ''.join(e)
            alphabet[a] = 0.0   
        return alphabet

    def count_Distribution(self, x:int , y:int, z:int):
        '''
        count the occurrences of  (x(t), y(t), y(t+τ)) 
            x affect y. For convenience, let X, Y, and Z denote x(t), y(t), y(t+τ)seperately.

        Parameters
        --
        x,y: int
            the index of vairable
        '''
        self.alphabet = self.want_An_Empty_Distribution()
        # count
        for i in range(self.numOfSteps-1):
            # index = str(int(self.bins[x][i])) + str(int(self.bins[y][i])) + str(int(self.bins[y][i+1]))
            index = self.alphabetBook[int(self.bins[x][i])] + self.alphabetBook[int(self.bins[y][i])] + self.alphabetBook[int(self.bins[z][i+1])]
            
            # print(index)
            self.alphabet[index] += 1/(self.numOfSteps-1)
        alphabetKeys  = list(self.alphabet.keys()) # seperate the keys form dict
        alphabetValue = list(self.alphabet.values()) # seperate the value from dict

        self.XYZ = dit.Distribution(alphabetKeys, alphabetValue) # get the joint distribution
        self.XYZ.set_rv_names('XYZ')
        pass     

    def mutual_Information(self,dis):
        '''
        calculate the mutual information of x(t) and y(t)

        Return
        ---
            dis: distribution
            mI: mutual information of x(t) and y(t)
        '''
        
        mI = dit.shannon.mutual_information(dis,'X','Y')
        # print(mutual_Information)
        return mI
        

    def time_Delayed_Mutual_Information(self,dis):
        '''
        calculate time delayed mutual information (TDMI)
        
        Return
        ---
            tMDI: TMDI
        '''
        tMDI = dit.shannon.mutual_information(dis,'X','Z')
        return tMDI

    def transfer_Entropy(self,dis):
        '''
        calculate  transfer_Entropy (TE)
        
        Return
        ---
            tE: TE
        '''
        # tE = dit.multivariate.coinformation(self.XYZ, rvs = 'XZ', crvs = 'Y')
        tE = dit.multivariate.coinformation(dis, 'XZ','Y')
        return tE

    def intrinsic_Information_Flow(self,dis):
        '''
        calculate time intrinsic_Information_Flow (IIF)
        
        Return
        ---
            iIF: IIF
        '''
        try:
            iIF = dit.multivariate.secret_key_agreement.intrinsic_mutual_information(dis, 'XZ','Y')
        except:
            iIF=0
        return iIF

    # def shared_Information_Flow(self,  tE, iIF):
    def synergistic_Information_Flow(self,  tE, iIF):
        '''
        calculate time synergistic_Information_Flow (sYIF)
        
        Parameters
        ---
        tE: TE
        iIF: IIF

        Return
        ---
            sYIF: sYIF
        '''
        sYIF = tE - iIF
        return sYIF

    # def synergistic_Information_Flow(self, tDMI, iIF):
    def shared_Information_Flow(self, tDMI, iIF):
        '''
        calculate time shared_Information_Flow (SHIF)
        
        Parameters
        ---
        tDMI: TDMI
        iIF: IIF

        Return
        ---
            SHIF: SHIF
        '''
        sHIF = tDMI - iIF
        return sHIF

    def partial_Information(self, dis):
        '''
        calculate the partial information  
        
        '''
        d = dit.pid.PID_WB
        pass

    def calculate_Information(self):
        '''
        calculate all the information we need
        '''
        self.mI = self.mutual_Information(self.XYZ)
        self.tDMI = self.time_Delayed_Mutual_Information(self.XYZ)
        self.tE = self.transfer_Entropy(self.XYZ)
        self.iIF = self.intrinsic_Information_Flow(self.XYZ)
        self.sHIF = self.shared_Information_Flow(tDMI=self.tDMI, iIF=self.iIF)
        self.sYIF = self.synergistic_Information_Flow(tE= self.tE, iIF= self.iIF)
        pass

    def get_Information(self):
        '''
        Collect all the quantities we need in a `dict`.
        '''
        informationDict = {
            "mutual_information": self.mI,
            "time_delayed_mutual_information": self.tDMI,
            "transfer_entropy":self.tE,
            "intrinsic_information_flow":self.iIF,
            "shared_information_flow":self.sHIF,
            "synergistic_information_flow":self.sYIF,
        }

        return informationDict
    

    '''
    information consistency
    '''


    def count_Distribution_Adjacent(self, adjacentMatrix: np.ndarray):
        '''
        count distribution when they are neighbour

        parameters
        ---
        adjacentMatrix: adjacentMatrix
        '''
        x = self.x
        y = self.y
        z = self.z
        l = self.numOfSteps
        #
        self.alphabet = self.want_An_Empty_Distribution()
        sumOfNeighbour = np.sum(adjacentMatrix[x,y,:])
        sumNum = 0
        # threshold = 46656
        # threshold = 20000
        # threshold = 50000
        threshold = 0
        if sumOfNeighbour <= threshold :
            print(threshold,sumOfNeighbour," x=",x," y=",y)
            return False
        
        index = ''
        for i in range(l-1):
            # if adjacentMatrix[x][y][i] == 1 and adjacentMatrix[x][y][i+1] == 1: 
            if adjacentMatrix[x][z][i] == 1 and adjacentMatrix[y][z][i]==1: 
                index = self.alphabetBook[int(self.bins[x][i])] + self.alphabetBook[int(self.bins[y][i])] + self.alphabetBook[int(self.bins[z][i+1])]

                self.alphabet[index] += 1
                sumNum += 1
        for i in self.alphabet:       
            self.alphabet[i] = self.alphabet[i]/sumNum
        print(x,y,z,"ok",sumOfNeighbour)

        alphabetKeys  = list(self.alphabet.keys()) # seperate the keys form dict
        alphabetValue = list(self.alphabet.values()) # seperate the value from dict

        self.XYZ = dit.Distribution(alphabetKeys, alphabetValue) # get the joint distribution
        self.XYZ.set_rv_names('XYZ')


        return True
    